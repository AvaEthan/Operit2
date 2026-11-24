use std::collections::BTreeMap;

use crate::sqliteParams;
use crate::SqliteStore::{toSqliteValue, SqliteRow, SqliteRowGet, SqliteStore, SqliteStoreError, SqliteTransaction};

use operit_model::MessagePart::MessagePartKind;
use operit_model::MessagePartCodec::is_tool_result_blob_hash;
use operit_model::MessagePartEntity::MessagePartEntity;
use sha2::{Digest, Sha256};

const SELECT_PART_COLUMNS: &str = r#"
    SELECT chatId, messageTimestamp, variantIndex, partId, sequence, kind, content,
        toolCallId, toolName, attributesJson
    FROM message_parts
"#;
const PENDING_TOOL_RESULT_TTL_MILLIS: i64 = 24 * 60 * 60 * 1000;

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
                &[],
                &[],
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
        participantIds: &[String],
        extensionOwners: &[String],
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
        for mut part in parts {
            if part.kind == MessagePartKind::ToolResult && !part.content.is_empty() {
                let hash = toolResultBlobHash(&part.content);
                upsertToolResultBlobInTransaction(transaction, &part.chatId, &hash, &part.content)?;
                part.attributes.insert("blob_hash".into(), hash);
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
            if part.kind == MessagePartKind::ToolResult {
                if let Some(hash) = part.attributes.get("blob_hash") {
                    let placeholders = participantIds.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                    let ownerPlaceholders = extensionOwners.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                    let sql = if extensionOwners.is_empty() {
                        "DELETE FROM pending_tool_result_blobs WHERE hash = ?1 AND chatId = ?2 AND participantId IS NULL AND extensionOwner = ''".to_string()
                    } else {
                        format!("DELETE FROM pending_tool_result_blobs WHERE hash = ?1 AND chatId = ?2 AND (participantId IS NULL OR (extensionOwner IN ({ownerPlaceholders}) AND participantId IN ({placeholders})))")
                    };
                    let mut params = sqliteParams![hash, chatId];
                    params.extend(extensionOwners.iter().map(toSqliteValue));
                    params.extend(participantIds.iter().map(toSqliteValue));
                    transaction.execute(&sql, params)?;
                }
            }
        }
        Ok(())
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

    pub fn storeToolResultBlob(
        &self,
        chatId: &str,
        content: &str,
    ) -> Result<String, SqliteStoreError> {
        self.stageToolResultBlob(chatId, None, "", content)
    }

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
            let needsPending = participantId.is_some()
                || transaction
                    .queryOne(
                        r#"SELECT 1 FROM message_parts
                           WHERE chatId = ?1 AND kind = 'tool_result'
                             AND json_valid(attributesJson)
                             AND json_extract(attributesJson, '$.blob_hash') = ?2
                           LIMIT 1"#,
                        sqliteParams![chatId, hash],
                    )?
                    .is_none();
            if needsPending {
                transaction.execute(
                    r#"INSERT INTO pending_tool_result_blobs
                       (hash, chatId, participantId, participantKey, extensionOwner, created_at)
                       VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                       ON CONFLICT(hash, chatId, participantKey, extensionOwner) DO UPDATE SET
                         participantId = excluded.participantId,
                         extensionOwner = excluded.extensionOwner,
                         created_at = excluded.created_at"#,
                    sqliteParams![
                        hash,
                        chatId,
                        participantId,
                        participantId.unwrap_or_default(),
                        extensionOwner,
                        chrono::Utc::now().timestamp_millis()
                    ],
                )?;
            }
            Ok(hash)
        })
    }

    pub fn readToolResultBlob(
        &self,
        chatId: &str,
        participantId: Option<&str>,
        extensionOwner: &str,
        hash: &str,
        offset: usize,
        limit: usize,
    ) -> Result<String, SqliteStoreError> {
        if !is_tool_result_blob_hash(hash) {
            return Err(SqliteStoreError::Message(
                "Invalid tool result blob hash".into(),
            ));
        }
        let references = self.store.queryRows(
            r#"SELECT p.kind, p.content, COALESCE(v.pluginExtensions, m.pluginExtensions) AS pluginExtensions
               FROM message_parts p
               LEFT JOIN messages m ON m.chatId = p.chatId AND m.timestamp = p.messageTimestamp
               LEFT JOIN message_variants v
                 ON v.chatId = p.chatId AND v.messageTimestamp = p.messageTimestamp
                AND v.variantIndex = p.variantIndex
               WHERE p.chatId = ?1 AND json_valid(p.attributesJson)
                 AND json_extract(p.attributesJson, '$.blob_hash') = ?2"#,
            sqliteParams![chatId, hash],
        )?;
        let authorizedReference = references.iter().find(|row| {
            let isMatchingResult = row
                .get::<_, String>("kind")
                .ok()
                .and_then(|kind| partKindFromLabel(&kind).ok())
                == Some(MessagePartKind::ToolResult)
                && row
                    .get::<_, String>("content")
                    .ok()
                    .is_some_and(|content| toolResultBlobHash(&content) == hash);
            if !isMatchingResult {
                return false;
            }
            let Some(participantId) = participantId else {
                return true;
            };
            row.get::<_, String>("pluginExtensions")
                .ok()
                .and_then(|json| {
                    serde_json::from_str::<BTreeMap<String, serde_json::Value>>(json.as_str()).ok()
                })
                .and_then(|identities| identities.get(extensionOwner).cloned())
                .and_then(|identity| identity.get("participantId").cloned())
                .and_then(|participant| participant.as_str().map(str::to_string))
                .is_some_and(|author| author == participantId)
        });
        let pendingReference = self
            .store
            .queryOne(
                "SELECT 1 FROM pending_tool_result_blobs WHERE hash = ?1 AND chatId = ?2 AND (?3 IS NULL OR participantId = ?3) AND extensionOwner = ?4 AND created_at >= ?5 LIMIT 1",
                sqliteParams![hash, chatId, participantId, extensionOwner, pendingToolResultExpiryCutoff()],
            )?
            .is_some();
        if authorizedReference.is_none() && !pendingReference {
            return Err(SqliteStoreError::Message(format!(
                "Tool result blob {hash} is not referenced by chat {chatId} for this participant"
            )));
        }
        let localContent: Option<String> = self
            .store
            .queryOne(
                "SELECT content FROM tool_result_blobs WHERE hash = ?1",
                sqliteParams![hash],
            )?
            .map(|row| row.get("content"))
            .transpose()?;
        let content = match localContent {
            Some(content) => content,
            None => authorizedReference
                .map(|row| row.get::<_, String>("content"))
                .transpose()?
                .ok_or_else(|| SqliteStoreError::Message(format!(
                    "Tool result blob {hash} is unavailable on this device; rerun the original tool to recreate its full result"
                )))?,
        };
        if toolResultBlobHash(&content) != hash {
            return Err(SqliteStoreError::Message(format!(
                "Tool result blob {hash} content does not match its hash"
            )));
        }
        Ok(content.chars().skip(offset).take(limit).collect())
    }

    pub fn collectUnreferencedToolResultBlobs(&self) -> Result<(), SqliteStoreError> {
        self.store
            .transaction(|transaction| collectUnreferencedToolResultBlobsInTransaction(transaction))
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

pub(crate) fn normalizeToolResultBlobsInTransaction(
    transaction: &mut SqliteTransaction<'_>,
    parts: &mut [MessagePartEntity],
) -> Result<(), SqliteStoreError> {
    for part in parts {
        if part.kind != MessagePartKind::ToolResult || part.content.is_empty() {
            continue;
        }
        let hash = toolResultBlobHash(&part.content);
        if part.attributes.get("blob_hash") == Some(&hash) {
            continue;
        }
        upsertToolResultBlobInTransaction(transaction, &part.chatId, &hash, &part.content)?;
        part.attributes.insert("blob_hash".into(), hash.clone());
        let attributesJson = serde_json::to_string(&part.attributes).map_err(|error| {
            SqliteStoreError::Message(format!("message-part attributes cannot serialize: {error}"))
        })?;
        transaction.execute(
            "UPDATE message_parts SET attributesJson = ?1 WHERE chatId = ?2 AND messageTimestamp = ?3 AND variantIndex = ?4 AND partId = ?5",
            sqliteParams![
                attributesJson,
                part.chatId,
                part.messageTimestamp,
                part.variantIndex,
                part.partId
            ],
        )?;
    }
    Ok(())
}

pub(crate) fn upsertToolResultBlobInTransaction(
    transaction: &mut SqliteTransaction<'_>,
    chatId: &str,
    hash: &str,
    content: &str,
) -> Result<(), SqliteStoreError> {
    transaction.execute(
        r#"INSERT INTO tool_result_blobs (hash, chatId, bytes, created_at, content)
           VALUES (?1, ?2, ?3, ?4, ?5)
           ON CONFLICT(hash) DO NOTHING"#,
        sqliteParams![
            hash,
            chatId,
            content.len() as i64,
            chrono::Utc::now().timestamp_millis(),
            content
        ],
    )?;
    Ok(())
}

pub(crate) fn collectUnreferencedToolResultBlobsInTransaction(
    transaction: &mut SqliteTransaction<'_>,
) -> Result<(), SqliteStoreError> {
    transaction.execute(
        "DELETE FROM pending_tool_result_blobs WHERE created_at < ?1",
        sqliteParams![pendingToolResultExpiryCutoff()],
    )?;
    transaction.execute(
        r#"DELETE FROM tool_result_blobs
           WHERE NOT EXISTS (
               SELECT 1 FROM message_parts
               WHERE json_valid(message_parts.attributesJson)
                 AND json_extract(message_parts.attributesJson, '$.blob_hash') = tool_result_blobs.hash
           )
           AND NOT EXISTS (
               SELECT 1 FROM pending_tool_result_blobs
               WHERE pending_tool_result_blobs.hash = tool_result_blobs.hash
           )"#,
        Vec::new(),
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

pub fn toolResultBlobHash(content: &str) -> String {
    format!("{:x}", Sha256::digest(content.as_bytes()))
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
