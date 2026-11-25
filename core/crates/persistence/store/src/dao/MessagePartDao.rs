use std::fmt::Write;

use crate::sqliteParams;
use crate::SqliteStore::{toSqliteValue, SqliteRow, SqliteRowGet, SqliteStore, SqliteStoreError, SqliteTransaction};

use operit_model::MessagePart::MessagePartKind;
use operit_model::MessagePartEntity::MessagePartEntity;
use sha2::{Digest, Sha256};

const SELECT_PART_COLUMNS: &str = r#"
    SELECT chatId, messageTimestamp, variantIndex, partId, sequence, kind, content,
        toolCallId, toolName, attributesJson
    FROM message_parts
"#;

/// Attribute that records where a tool result's full payload is kept.
pub const TOOL_RESULT_BLOB_HASH_ATTRIBUTE: &str = "blob_hash";
/// Staged results are unreachable until their part is written, so they expire instead of leaking.
const PENDING_TOOL_RESULT_TTL_MILLIS: i64 = 24 * 60 * 60 * 1000;

/// Computes the content address that identifies one tool result payload.
pub fn toolResultBlobHash(content: &str) -> String {
    let digest = Sha256::digest(content.as_bytes());
    let mut hash = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut hash, "{byte:02x}").expect("hex writing into a String cannot fail");
    }
    hash
}

fn pendingToolResultExpiryCutoff() -> i64 {
    chrono::Utc::now().timestamp_millis() - PENDING_TOOL_RESULT_TTL_MILLIS
}

/// Provides persistence operations for ordered structured message parts.
#[derive(Clone)]
pub struct MessagePartDao {
    store: SqliteStore,
}

impl MessagePartDao {
    /// Creates a part DAO bound to one SQLite store.
    pub fn new(store: SqliteStore) -> Self {
        Self { store }
    }

    /// Loads all parts for one message revision in display order.
    pub fn getPartsForMessage(
        &self,
        chatId: &str,
        messageTimestamp: i64,
        variantIndex: i32,
    ) -> Result<Vec<MessagePartEntity>, SqliteStoreError> {
        self.selectParts(
            &format!(
                "{SELECT_PART_COLUMNS}
                WHERE chatId = ?1 AND messageTimestamp = ?2 AND variantIndex = ?3
                ORDER BY sequence ASC"
            ),
            sqliteParams![chatId, messageTimestamp, variantIndex],
        )
    }

    /// Loads all parts for an explicitly requested full-chat snapshot.
    pub fn getPartsForChat(
        &self,
        chatId: &str,
    ) -> Result<Vec<MessagePartEntity>, SqliteStoreError> {
        self.selectParts(
            &format!(
                "{SELECT_PART_COLUMNS}
                WHERE chatId = ?1
                ORDER BY messageTimestamp ASC, variantIndex ASC, sequence ASC"
            ),
            sqliteParams![chatId],
        )
    }

    /// Loads all revisions' parts for the supplied message timestamps in display order.
    pub fn getPartsForMessages(
        &self,
        chatId: &str,
        messageTimestamps: Vec<i64>,
    ) -> Result<Vec<MessagePartEntity>, SqliteStoreError> {
        let placeholders = messageTimestamps
            .iter()
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "{SELECT_PART_COLUMNS}
            WHERE chatId = ? AND messageTimestamp IN ({placeholders})
            ORDER BY messageTimestamp ASC, variantIndex ASC, sequence ASC"
        );
        let mut params = sqliteParams![chatId];
        for timestamp in &messageTimestamps {
            params.push(toSqliteValue(timestamp));
        }
        self.selectParts(&sql, params)
    }

    /// Replaces the complete ordered part set for one message revision atomically.
    pub fn replaceParts(
        &self,
        chatId: &str,
        messageTimestamp: i64,
        variantIndex: i32,
        parts: Vec<MessagePartEntity>,
    ) -> Result<(), SqliteStoreError> {
        self.store.transaction(|transaction| {
            Self::replacePartsInTransaction(
                transaction,
                chatId,
                messageTimestamp,
                variantIndex,
                parts,
            )
        })
    }

    /// Replaces parts inside the transaction that also writes the owning message revision.
    pub(crate) fn replacePartsInTransaction(
        transaction: &mut SqliteTransaction<'_>,
        chatId: &str,
        messageTimestamp: i64,
        variantIndex: i32,
        parts: Vec<MessagePartEntity>,
    ) -> Result<(), SqliteStoreError> {
        if parts.is_empty() {
            return Err(SqliteStoreError::Message(format!(
                "Message {chatId}:{messageTimestamp} revision {variantIndex} requires at least one part"
            )));
        }
        transaction.execute(
            "DELETE FROM message_parts WHERE chatId = ?1 AND messageTimestamp = ?2 AND variantIndex = ?3",
            sqliteParams![chatId, messageTimestamp, variantIndex],
        )?;
        let mut blobHashes: Vec<String> = Vec::new();
        for mut part in parts {
            if part.kind == MessagePartKind::ToolResult && !part.content.is_empty() {
                let hash = toolResultBlobHash(&part.content);
                upsertToolResultBlobInTransaction(transaction, &part.chatId, &hash, &part.content)?;
                part.attributes
                    .insert(TOOL_RESULT_BLOB_HASH_ATTRIBUTE.into(), hash.clone());
                blobHashes.push(hash);
            }
            transaction.execute(
                r#"
                INSERT INTO message_parts (
                    chatId, messageTimestamp, variantIndex, partId, sequence, kind, content,
                    toolCallId, toolName, attributesJson
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                "#,
                partParams(&part)?,
            )?;
        }
        if !blobHashes.is_empty() {
            let placeholders = blobHashes.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let mut params = sqliteParams![chatId];
            params.extend(blobHashes.iter().map(toSqliteValue));
            transaction.execute(
                &format!(
                    "DELETE FROM pending_tool_result_blobs WHERE chatId = ?1 AND hash IN ({placeholders})"
                ),
                params,
            )?;
        }
        Ok(())
    }

    /// Keeps one tool payload addressable before its part row exists, for the extension that produced it.
    pub fn stageToolResultBlob(
        &self,
        chatId: &str,
        participantId: Option<&str>,
        extensionOwner: &str,
        content: &str,
    ) -> Result<String, SqliteStoreError> {
        let hash = toolResultBlobHash(content);
        self.store.transaction(|transaction| {
            upsertToolResultBlobInTransaction(transaction, chatId, &hash, content)?;
            transaction.execute(
                r#"INSERT INTO pending_tool_result_blobs
                   (chatId, hash, participantId, participantKey, extensionOwner, created_at)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                   ON CONFLICT(chatId, hash, participantKey, extensionOwner)
                   DO UPDATE SET created_at = excluded.created_at"#,
                sqliteParams![
                    chatId,
                    hash,
                    participantId,
                    participantId.unwrap_or_default(),
                    extensionOwner,
                    chrono::Utc::now().timestamp_millis()
                ],
            )?;
            Ok(hash)
        })
    }

    /// Reads one character slice of an addressed tool payload that this chat may still reference.
    ///
    /// The payload must either belong to a part that is already written in this chat or still be
    /// staged by the very participant that asks, so another chat cannot address it by hash.
    pub fn readToolResultBlob(
        &self,
        chatId: &str,
        participantId: Option<&str>,
        extensionOwner: &str,
        hash: &str,
        offset: usize,
        limit: usize,
    ) -> Result<String, SqliteStoreError> {
        if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(SqliteStoreError::Message(
                "Tool result blob hash must be 64 hexadecimal characters".into(),
            ));
        }
        let hash = hash.to_ascii_lowercase();
        let referenced = self
            .store
            .queryOne(
                r#"SELECT 1 FROM message_parts
                   WHERE chatId = ?1 AND kind = 'tool_result'
                     AND json_valid(attributesJson)
                     AND json_extract(attributesJson, '$.blob_hash') = ?2
                   LIMIT 1"#,
                sqliteParams![chatId, hash],
            )?
            .is_some();
        let staged = !referenced
            && self
                .store
                .queryOne(
                    r#"SELECT 1 FROM pending_tool_result_blobs
                       WHERE chatId = ?1 AND hash = ?2 AND participantKey = ?3
                         AND extensionOwner = ?4 AND created_at >= ?5
                       LIMIT 1"#,
                    sqliteParams![
                        chatId,
                        hash,
                        participantId.unwrap_or_default(),
                        extensionOwner,
                        pendingToolResultExpiryCutoff()
                    ],
                )?
                .is_some();
        if !referenced && !staged {
            return Err(SqliteStoreError::Message(format!(
                "Tool result blob {hash} is not referenced by chat {chatId}"
            )));
        }
        let localContent: Option<String> = self
            .store
            .queryOne(
                "SELECT content FROM tool_result_blobs WHERE chatId = ?1 AND hash = ?2",
                sqliteParams![chatId, hash],
            )?
            .map(|row| row.get("content"))
            .transpose()?;
        let transcriptContent = if localContent.is_none() && referenced {
            self.store
                .queryOne(
                    r#"SELECT content FROM message_parts
                       WHERE chatId = ?1 AND kind = 'tool_result'
                         AND json_valid(attributesJson)
                         AND json_extract(attributesJson, '$.blob_hash') = ?2
                       LIMIT 1"#,
                    sqliteParams![chatId, hash],
                )?
                .map(|row| row.get("content"))
                .transpose()?
        } else {
            None
        };
        let content = localContent.or(transcriptContent).ok_or_else(|| {
            SqliteStoreError::Message(format!(
                "Tool result blob {hash} is unavailable on this device; rerun the original tool to recreate its full result"
            ))
        })?;
        if toolResultBlobHash(&content) != hash {
            return Err(SqliteStoreError::Message(format!(
                "Tool result blob {hash} content does not match its hash"
            )));
        }
        Ok(content.chars().skip(offset).take(limit).collect())
    }

    /// Drops staged payloads that never reached a part row, then the stored copies nobody references.
    pub fn purgeExpiredToolResultBlobs(&self) -> Result<usize, SqliteStoreError> {
        let cutoff = pendingToolResultExpiryCutoff();
        self.store.transaction(|transaction| {
            let expiredPending = transaction.execute(
                "DELETE FROM pending_tool_result_blobs WHERE created_at < ?1",
                sqliteParams![cutoff],
            )?;
            transaction.execute(
                &format!(
                    r#"DELETE FROM tool_result_blobs
                       WHERE NOT EXISTS (
                           SELECT 1 FROM message_parts
                           WHERE message_parts.chatId = tool_result_blobs.chatId
                             AND message_parts.kind = 'tool_result'
                             AND json_valid(message_parts.attributesJson)
                             AND json_extract(message_parts.attributesJson, '$.{TOOL_RESULT_BLOB_HASH_ATTRIBUTE}') = tool_result_blobs.hash
                       )
                       AND NOT EXISTS (
                           SELECT 1 FROM pending_tool_result_blobs AS live
                           WHERE live.chatId = tool_result_blobs.chatId
                             AND live.hash = tool_result_blobs.hash
                       )"#
                ),
                Vec::new(),
            )?;
            Ok(expiredPending)
        })
    }

    /// Copies all selected revision parts into a different chat identifier.
    pub fn copyPartsToChat(
        &self,
        sourceChatId: &str,
        targetChatId: &str,
        upToTimestampInclusive: Option<i64>,
    ) -> Result<(), SqliteStoreError> {
        self.store.execute(
            r#"
            INSERT INTO message_parts (
                chatId, messageTimestamp, variantIndex, partId, sequence, kind, content,
                toolCallId, toolName, attributesJson
            )
            SELECT
                ?2, messageTimestamp, variantIndex, partId, sequence, kind, content,
                toolCallId, toolName, attributesJson
            FROM message_parts
            WHERE chatId = ?1 AND (?3 IS NULL OR messageTimestamp <= ?3)
            "#,
            sqliteParams![sourceChatId, targetChatId, upToTimestampInclusive],
        )?;
        Ok(())
    }

    /// Deletes all parts belonging to one message revision.
    pub fn deletePartsForMessage(
        &self,
        chatId: &str,
        messageTimestamp: i64,
        variantIndex: i32,
    ) -> Result<(), SqliteStoreError> {
        self.store.execute(
            "DELETE FROM message_parts WHERE chatId = ?1 AND messageTimestamp = ?2 AND variantIndex = ?3",
            sqliteParams![chatId, messageTimestamp, variantIndex],
        )?;
        Ok(())
    }

    /// Deletes all parts belonging to all revisions of one message.
    pub fn deletePartsForMessageTimestamp(
        &self,
        chatId: &str,
        messageTimestamp: i64,
    ) -> Result<(), SqliteStoreError> {
        self.store.execute(
            "DELETE FROM message_parts WHERE chatId = ?1 AND messageTimestamp = ?2",
            sqliteParams![chatId, messageTimestamp],
        )?;
        Ok(())
    }

    /// Deletes parts from one timestamp through the end of a chat.
    pub fn deletePartsFrom(
        &self,
        chatId: &str,
        messageTimestamp: i64,
    ) -> Result<(), SqliteStoreError> {
        self.store.execute(
            "DELETE FROM message_parts WHERE chatId = ?1 AND messageTimestamp >= ?2",
            sqliteParams![chatId, messageTimestamp],
        )?;
        Ok(())
    }

    /// Deletes every structured part associated with a chat.
    pub fn deleteAllPartsForChat(&self, chatId: &str) -> Result<(), SqliteStoreError> {
        self.store.execute(
            "DELETE FROM message_parts WHERE chatId = ?1",
            sqliteParams![chatId],
        )?;
        Ok(())
    }

    /// Loads part rows with the caller's ordering and filtering statement.
    fn selectParts(
        &self,
        sql: &str,
        params: Vec<operit_host_api::SqliteValue>,
    ) -> Result<Vec<MessagePartEntity>, SqliteStoreError> {
        self.store
            .queryRows(sql, params)?
            .into_iter()
            .map(|row| mapMessagePartEntity(&row))
            .collect()
    }
}

/// Converts one SQLite row into a typed message part entity.
pub(crate) fn mapMessagePartEntity(row: &SqliteRow) -> Result<MessagePartEntity, SqliteStoreError> {
    let attributesJson: String = row.get("attributesJson")?;
    let kind: String = row.get("kind")?;
    Ok(MessagePartEntity {
        chatId: row.get("chatId")?,
        messageTimestamp: row.get("messageTimestamp")?,
        variantIndex: row.get("variantIndex")?,
        partId: row.get("partId")?,
        sequence: row.get("sequence")?,
        kind: partKindFromLabel(&kind)?,
        content: row.get("content")?,
        toolCallId: row.get("toolCallId")?,
        toolName: row.get("toolName")?,
        attributes: serde_json::from_str(&attributesJson).map_err(|error| {
            SqliteStoreError::Message(format!("invalid message-part attributes JSON: {error}"))
        })?,
    })
}

/// Keeps one payload under its content address for the chat that produced it.
///
/// The first writer wins: identical content in the same chat is stored once.
pub(crate) fn upsertToolResultBlobInTransaction(
    transaction: &mut SqliteTransaction<'_>,
    chatId: &str,
    hash: &str,
    content: &str,
) -> Result<(), SqliteStoreError> {
    transaction.execute(
        r#"INSERT INTO tool_result_blobs (chatId, hash, bytes, created_at, content)
           VALUES (?1, ?2, ?3, ?4, ?5)
           ON CONFLICT(chatId, hash) DO NOTHING"#,
        sqliteParams![
            chatId,
            hash,
            content.len() as i64,
            chrono::Utc::now().timestamp_millis(),
            content
        ],
    )?;
    Ok(())
}

/// Serializes a message part entity for one INSERT statement.
pub(crate) fn partParams(
    part: &MessagePartEntity,
) -> Result<Vec<operit_host_api::SqliteValue>, SqliteStoreError> {
    let attributesJson = serde_json::to_string(&part.attributes).map_err(|error| {
        SqliteStoreError::Message(format!("message-part attributes cannot serialize: {error}"))
    })?;
    Ok(sqliteParams![
        part.chatId,
        part.messageTimestamp,
        part.variantIndex,
        part.partId,
        part.sequence,
        partKindLabel(&part.kind),
        part.content,
        part.toolCallId,
        part.toolName,
        attributesJson,
    ])
}

/// Converts a message-part kind into its database label.
fn partKindLabel(kind: &MessagePartKind) -> &'static str {
    match kind {
        MessagePartKind::Markdown => "markdown",
        MessagePartKind::Thinking => "thinking",
        MessagePartKind::ToolCall => "tool_call",
        MessagePartKind::ToolResult => "tool_result",
        MessagePartKind::Status => "status",
    }
}

/// Converts a database label into a message-part kind.
fn partKindFromLabel(value: &str) -> Result<MessagePartKind, SqliteStoreError> {
    match value {
        "markdown" => Ok(MessagePartKind::Markdown),
        "thinking" => Ok(MessagePartKind::Thinking),
        "tool_call" => Ok(MessagePartKind::ToolCall),
        "tool_result" => Ok(MessagePartKind::ToolResult),
        "status" => Ok(MessagePartKind::Status),
        _ => Err(SqliteStoreError::Message(format!(
            "unknown message-part kind: {value}"
        ))),
    }
}

#[cfg(test)]
#[path = "MessagePartDaoTests.rs"]
mod blob_tests;
