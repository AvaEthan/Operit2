use super::*;

use crate::sync::SqlChatSyncStore::tests::{openTestStore, DATABASE_MUTEX};
use operit_model::ChatEntity::ChatEntity;
use operit_model::MessagePart::MessagePart;

const CHAT_ID: &str = "chat-tool-blobs";
const OWNER: &str = "com.operit.test";

/// Writes one tool result part for the supplied payload.
fn toolResultParts(content: &str) -> Vec<MessagePartEntity> {
    vec![MessagePartEntity::fromMessagePart(
        CHAT_ID.to_string(),
        100,
        0,
        MessagePart::toolResult(
            "part-0".to_string(),
            0,
            None,
            "read_file".to_string(),
            "success".to_string(),
            content.to_string(),
        ),
    )]
}

/// Opens a fresh database that already contains the fixture chat.
fn blobDatabase(name: &str) -> std::sync::Arc<crate::db::AppDatabase::AppDatabase> {
    let (_, database, _) = openTestStore(name);
    database
        .chatDao()
        .insertChat(ChatEntity::new(
            CHAT_ID.to_string(),
            "New Chat".to_string(),
            1_000,
        ))
        .unwrap();
    database
}

fn addressOfStoredResult(dao: &MessagePartDao, content: &str) -> String {
    dao.replaceParts(CHAT_ID, 100, 0, toolResultParts(content)).unwrap();
    dao.getPartsForMessage(CHAT_ID, 100, 0).unwrap()[0]
        .attributes
        .get(TOOL_RESULT_BLOB_HASH_ATTRIBUTE)
        .cloned()
        .unwrap_or_else(|| panic!("a stored tool result must carry its content address"))
}

#[test]
fn stored_tool_result_is_addressed_and_sliceable() {
    let _guard = DATABASE_MUTEX.lock().unwrap();
    let database = blobDatabase("blob-addressed-on-write");
    let dao = database.messagePartDao();
    let content = "x".repeat(4104);
    let hash = addressOfStoredResult(&dao, &content);
    assert_eq!(hash, toolResultBlobHash(&content));

    assert_eq!(
        dao.readToolResultBlob(CHAT_ID, None, OWNER, &hash, 0, content.chars().count()).unwrap(),
        content
    );
    assert_eq!(dao.readToolResultBlob(CHAT_ID, None, OWNER, &hash, 4, 3).unwrap(), "xxx");
    assert_eq!(dao.readToolResultBlob(CHAT_ID, None, OWNER, &hash, content.len(), 8).unwrap(), "");
}

#[test]
fn empty_tool_result_is_not_addressed() {
    let _guard = DATABASE_MUTEX.lock().unwrap();
    let database = blobDatabase("blob-empty-result");
    let dao = database.messagePartDao();
    dao.replaceParts(CHAT_ID, 100, 0, toolResultParts("")).unwrap();

    assert!(!dao.getPartsForMessage(CHAT_ID, 100, 0).unwrap()[0]
        .attributes
        .contains_key(TOOL_RESULT_BLOB_HASH_ATTRIBUTE));
}

#[test]
fn staged_result_is_readable_only_by_the_participant_that_staged_it() {
    let _guard = DATABASE_MUTEX.lock().unwrap();
    let database = blobDatabase("blob-staged-participant");
    let dao = database.messagePartDao();
    let hash = dao
        .stageToolResultBlob(CHAT_ID, Some("participant-a"), OWNER, "staged payload")
        .unwrap();

    assert_eq!(
        dao.readToolResultBlob(CHAT_ID, Some("participant-a"), OWNER, &hash, 0, 64).unwrap(),
        "staged payload"
    );
    assert!(dao
        .readToolResultBlob(CHAT_ID, Some("participant-b"), OWNER, &hash, 0, 64)
        .is_err());

    dao.replaceParts(CHAT_ID, 100, 0, toolResultParts("staged payload")).unwrap();
    assert_eq!(
        dao.readToolResultBlob(CHAT_ID, Some("participant-b"), OWNER, &hash, 0, 64).unwrap(),
        "staged payload"
    );
    assert_eq!(pendingRowCount(&database), 0);
}

#[test]
fn another_chat_cannot_address_the_hash() {
    let _guard = DATABASE_MUTEX.lock().unwrap();
    let database = blobDatabase("blob-other-chat");
    let dao = database.messagePartDao();
    let hash = dao
        .stageToolResultBlob(CHAT_ID, None, OWNER, "private payload")
        .unwrap();

    assert!(dao.readToolResultBlob("other-chat", None, OWNER, &hash, 0, 64).is_err());
    assert_eq!(dao.readToolResultBlob(CHAT_ID, None, OWNER, &hash, 0, 64).unwrap(), "private payload");
    assert!(dao.readToolResultBlob(CHAT_ID, None, OWNER, "not-a-hash", 0, 64).is_err());
}

#[test]
fn addressed_result_survives_a_missing_local_blob_cache() {
    let _guard = DATABASE_MUTEX.lock().unwrap();
    let database = blobDatabase("blob-cache-miss");
    let dao = database.messagePartDao();
    let content = "payload copied from another device";
    let hash = addressOfStoredResult(&dao, content);
    database
        .store()
        .execute("DELETE FROM tool_result_blobs", Vec::new())
        .unwrap();

    assert_eq!(
        dao.readToolResultBlob(
            CHAT_ID,
            None,
            OWNER,
            &hash.to_ascii_uppercase(),
            0,
            64,
        )
        .unwrap(),
        content
    );
}

#[test]
fn expiry_drops_staged_payloads_but_keeps_referenced_ones() {
    let _guard = DATABASE_MUTEX.lock().unwrap();
    let database = blobDatabase("blob-pending-expiry");
    let dao = database.messagePartDao();
    let staged = dao
        .stageToolResultBlob(CHAT_ID, None, OWNER, "abandoned payload")
        .unwrap();
    let referenced = addressOfStoredResult(&dao, "payload kept by its part");
    database
        .store()
        .execute("UPDATE pending_tool_result_blobs SET created_at = 0", Vec::new())
        .unwrap();

    assert!(dao.readToolResultBlob(CHAT_ID, None, OWNER, &staged, 0, 64).is_err());
    assert_eq!(dao.purgeExpiredToolResultBlobs().unwrap(), 1);
    assert_eq!(pendingRowCount(&database), 0);
    assert_eq!(
        dao.readToolResultBlob(CHAT_ID, None, OWNER, &referenced, 0, 64).unwrap(),
        "payload kept by its part"
    );
    assert_eq!(blobRowCount(&database), 1);
}

fn pendingRowCount(database: &std::sync::Arc<crate::db::AppDatabase::AppDatabase>) -> i64 {
    database
        .store()
        .queryScalar::<i64>("SELECT COUNT(*) FROM pending_tool_result_blobs", Vec::new())
        .unwrap()
}

fn blobRowCount(database: &std::sync::Arc<crate::db::AppDatabase::AppDatabase>) -> i64 {
    database
        .store()
        .queryScalar::<i64>("SELECT COUNT(*) FROM tool_result_blobs", Vec::new())
        .unwrap()
}
