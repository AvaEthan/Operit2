use crate::tools::ToolExecutionLimits::ToolExecutionLimits;
use operit_plugin_sdk::js_sdk::results::ToolResultData;
use operit_util::ChatMarkupRegex::ChatMarkupRegex;
use serde::{Deserialize, Serialize};

pub const ENHANCED_EMPTY_RESPONSE_WARNING: &str =
    "警告：本轮返回内容全部为空。请继续完成当前任务，并输出正文内容或调用所需工具。";
pub const ENHANCED_PURE_THINKING_ONLY_WARNING: &str = "警告：请输出正文内容，禁止仅输出思考内容。";
pub const ENHANCED_TRUNCATED_TOOL_CALL_WARNING: &str = "警告：检测到工具调用输出被截断。本轮所有工具调用均已作废且不会执行。请尝试减少单次输出、拆分任务，或更换更合适的模型/供应商后重试。";

#[derive(Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub toolName: String,
    pub success: bool,
    pub result: ToolResultData,
    pub error: Option<String>,
}

pub struct ConversationMarkupManager;

impl ConversationMarkupManager {
    /// Creates an explicit tool-error status message.
    pub fn createToolErrorStatus(toolName: &str, errorMessage: &str) -> String {
        Self::createToolResultXml(
            toolName,
            "error",
            &format!("<error>{errorMessage}</error>"),
        )
    }

    /// Creates a conversation warning without fabricating a tool result.
    pub fn createWarningStatus(warningMessage: &str) -> String {
        format!(r#"<status type="warning">{}</status>"#, warningMessage)
    }

    /// Formats one complete result for persistence without imposing a transport budget.
    pub fn formatToolResultForMessage(result: &ToolResult) -> String {
        Self::formatToolResultForMessageWithBlobHash(result, None)
    }

    pub fn formatToolResultForMessageWithBlobHash(
        result: &ToolResult,
        blobHash: Option<&str>,
    ) -> String {
        let markup = Self::createToolResultXml(
            &result.toolName,
            if result.success { "success" } else { "error" },
            &Self::toolResultPayload(result),
        );
        Self::withBlobHashAttribute(markup, blobHash)
    }

    /// Returns the exact full payload stored in a content-addressed blob.
    pub fn toolResultPayload(result: &ToolResult) -> String {
        if result.success {
            return result.result.toString();
        }
        let message = result.error.clone().unwrap_or_default().trim().to_string();
        let detail = result.result.toString().trim().to_string();
        if !message.is_empty() && !detail.is_empty() {
            format!("<error>{message}\n\n{detail}</error>")
        } else if !message.is_empty() {
            format!("<error>{message}</error>")
        } else {
            format!("<error>{detail}</error>")
        }
    }

    /// Adds structured UI metadata without changing the text-only model result contract.
    pub fn formatToolResultForDisplay(
        result: &ToolResult,
        display: Option<&operit_plugin_sdk::js_sdk::tool_display::ToolDisplayDeclaration>,
    ) -> String {
        Self::formatToolResultForDisplayWithBlobHash(result, display, None)
    }

    pub fn formatToolResultForDisplayWithBlobHash(
        result: &ToolResult,
        display: Option<&operit_plugin_sdk::js_sdk::tool_display::ToolDisplayDeclaration>,
        blobHash: Option<&str>,
    ) -> String {
        let mut markup = Self::formatToolResultForMessageWithBlobHash(result, blobHash);
        let mut attributes = String::new();
        let mut append = |name: &str, value: String| {
            let escaped = value
                .replace('&', "&amp;")
                .replace('"', "&quot;")
                .replace('<', "&lt;")
                .replace('>', "&gt;");
            attributes.push_str(&format!(" {name}=\"{escaped}\""));
        };
        if let Some(display) = display {
            append(
                "display",
                serde_json::to_string(display).expect("tool display serializes"),
            );
        }
        // Keep complete typed results only when they fit the existing per-result budget.
        // The text result remains available when a structured payload is too large.
        if display.is_some_and(|display| {
            display.result_renderer
                != operit_plugin_sdk::js_sdk::tool_display::ToolResultRenderer::Text
        }) {
            let data = match &result.result {
                ToolResultData::JsonResultData(data) => serde_json::to_string(&data.value),
                data => serde_json::to_string(data),
            }
            .expect("tool result serializes");
            if data.len() <= ToolExecutionLimits::MAX_SINGLE_TOOL_RESULT_MESSAGE_CHARS {
                append("result_data", data);
            }
        }
        if let Some(end) = markup.find('>') {
            markup.insert_str(end, &attributes);
        }
        markup
    }

    /// Bounds each result independently while preserving every result slot in invocation order.
    pub fn buildToolResultMessage(results: &[ToolResult]) -> String {
        Self::buildToolResultMessageWithBlobHashes(results, &[])
    }

    pub fn buildToolResultMessageWithBlobHashes(
        results: &[ToolResult],
        blobHashes: &[Option<String>],
    ) -> String {
        results
            .iter()
            .enumerate()
            .map(|(index, result)| {
                Self::formatToolResultForMessageWithBlobHash(
                    result,
                    blobHashes.get(index).and_then(Option::as_deref),
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn projectToolResultMessageForSend(
        message: &str,
        blobHashes: &[Option<String>],
        inlineLimit: Option<usize>,
    ) -> Result<String, String> {
        let Some(limit) = inlineLimit
            .filter(|limit| *limit > 0)
            .filter(|_| blobHashes.iter().any(Option::is_some))
        else {
            return Ok(message.to_string());
        };
        let parts = operit_model::MessagePartCodec::MessagePartCodec::parseAssistantMarkup(message)
            .map_err(|error| format!("Cannot project tool results for send: {error}"))?;
        Ok(
            operit_model::MessagePartCodec::MessagePartCodec::assistantMarkupWithToolResultLimit(
                &parts,
                Some(limit),
            ),
        )
    }

    /// Creates the error message for an unavailable tool.
    pub fn createToolNotAvailableError(toolName: &str, details: Option<&str>) -> String {
        let owned;
        let errorMessage = match details {
            Some(value) => value,
            None => {
                owned = format!("The tool `{}` is not available.", toolName);
                &owned
            }
        };
        Self::createToolErrorStatus(toolName, errorMessage)
    }

    /// Wraps a tool payload in one uniquely named result element.
    fn createToolResultXml(toolName: &str, status: &str, payload: &str) -> String {
        let tagName = ChatMarkupRegex::generate_random_tool_result_tag_name();
        format!(
            r#"<{tagName} name="{toolName}" status="{status}">{}</{tagName}>"#,
            operit_model::MessagePartCodec::toolResultContentMarkup(payload)
        )
    }

    fn withBlobHashAttribute(markup: String, blobHash: Option<&str>) -> String {
        let Some(hash) =
            blobHash.filter(|hash| operit_model::MessagePartCodec::is_tool_result_blob_hash(hash))
        else {
            return markup;
        };
        let Some(end) = markup.find('>') else {
            return markup;
        };
        let mut attributed = markup;
        attributed.insert_str(end, &format!(" blob_hash=\"{hash}\""));
        attributed
    }
}

#[cfg(test)]
mod tests {
    use operit_tools::tools::ToolResultDataClasses::{
        DirectoryListingData, FileEntry, StringResultData, TerminalCommandResultData,
        TerminalImplementation, TerminalType, ToolResultData,
    };
    use operit_tools::ConversationMarkupManager::{ConversationMarkupManager, ToolResult};

    #[test]
    fn display_metadata_preserves_native_result_type_and_stays_out_of_provider_text() {
        use operit_model::MessagePartCodec::MessagePartCodec;
        use operit_plugin_sdk::js_sdk::tool_display::ToolDisplayDeclaration;
        let result = ToolResult {
            toolName: "example:z42".into(),
            success: true,
            error: None,
            result: ToolResultData::DirectoryListingData(DirectoryListingData {
                path: "/remote/a&b".into(),
                entries: vec![],
            }),
        };
        let display = ToolDisplayDeclaration {
            label: std::collections::BTreeMap::from([("fr".into(), "Lire & \"voir\"".into())]),
            icon: "folder".into(),
            target_parameter: Some("document".into()),
            result_renderer: operit_plugin_sdk::js_sdk::tool_display::ToolResultRenderer::Directory,
        };
        let markup = ConversationMarkupManager::formatToolResultForDisplay(&result, Some(&display));
        let parts = MessagePartCodec::parseAssistantMarkup(&markup).unwrap();
        let part = parts
            .iter()
            .find(|part| part.kind == operit_model::MessagePart::MessagePartKind::ToolResult)
            .expect(&markup);
        assert!(
            part.attributes.contains_key("display"),
            "{markup} {:?}",
            part.attributes
        );
        assert_eq!(
            serde_json::from_str::<ToolDisplayDeclaration>(&part.attributes["display"]).unwrap(),
            display
        );
        let data: serde_json::Value =
            serde_json::from_str(&part.attributes["result_data"]).unwrap();
        assert_eq!(data["__type"], "DirectoryListingData");
        assert_eq!(data["path"], "/remote/a&b");
        assert_eq!(data["entries"], serde_json::json!([]));
        assert!(!MessagePartCodec::assistantMarkup(&parts).contains("result_data="));
        assert!(!MessagePartCodec::assistantMarkup(&parts).contains("display="));
        assert!(
            !ConversationMarkupManager::formatToolResultForMessage(&result)
                .contains("result_data=")
        );
        let stored = serde_json::to_string(&parts).unwrap();
        let restored: Vec<operit_model::MessagePart::MessagePart> =
            serde_json::from_str(&stored).unwrap();
        assert_eq!(restored, parts);
    }

    #[test]
    fn plugin_owned_json_uses_its_declared_renderer_without_native_type_inference() {
        use operit_plugin_sdk::js_sdk::tool_display::{ToolDisplayDeclaration, ToolResultRenderer};
        let payload = serde_json::json!({"path":"/plugin-owned", "entries":[]});
        let result = ToolResult {
            toolName: "example:unrelated_name".into(),
            success: true,
            error: None,
            result: ToolResultData::JsonResultData(
                operit_plugin_sdk::js_sdk::results::JsonResultData {
                    value: payload.clone(),
                },
            ),
        };
        assert!(
            !ConversationMarkupManager::formatToolResultForDisplay(&result, None)
                .contains("result_data=")
        );
        let mut display = ToolDisplayDeclaration {
            label: std::collections::BTreeMap::from([("en".into(), "View".into())]),
            ..Default::default()
        };
        assert!(
            !ConversationMarkupManager::formatToolResultForDisplay(&result, Some(&display))
                .contains("result_data=")
        );
        display.result_renderer = ToolResultRenderer::Directory;
        let markup = ConversationMarkupManager::formatToolResultForDisplay(&result, Some(&display));
        let parts = operit_model::MessagePartCodec::MessagePartCodec::parseAssistantMarkup(&markup)
            .unwrap();
        let part = parts
            .iter()
            .find(|part| part.kind == operit_model::MessagePart::MessagePartKind::ToolResult)
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&part.attributes["result_data"]).unwrap(),
            payload
        );
    }

    #[test]
    fn formats_runtime_tool_result_data_with_to_string() {
        let result = ToolResult {
            toolName: "execute_in_terminal_session".to_string(),
            success: true,
            result: ToolResultData::TerminalCommandResultData(TerminalCommandResultData {
                command: "Write-Output direct-package-ok".to_string(),
                output: "direct-package-ok\n".to_string(),
                exitCode: 0,
                sessionId: "session-1".to_string(),
                platform: "windows".to_string(),
                terminal: TerminalImplementation::Native,
                terminalType: TerminalType::Powershell,
                timedOut: false,
            }),
            error: None,
        };

        let formatted = ConversationMarkupManager::formatToolResultForMessage(&result);

        assert!(formatted.contains("Terminal Command Execution Result:"));
        assert!(formatted.contains("Command: Write-Output direct-package-ok"));
        assert!(formatted.contains("Session: session-1"));
        assert!(formatted.contains("Output:"));
        assert!(!formatted.contains(r#""__type":"TerminalCommandResultData""#));
    }

    #[test]
    fn formats_file_collection_tool_result_data_with_to_string() {
        let result = ToolResult {
            toolName: "list_files".to_string(),
            success: true,
            result: ToolResultData::DirectoryListingData(DirectoryListingData {
                path: "C:\\work".to_string(),
                entries: vec![FileEntry {
                    name: "notes.txt".to_string(),
                    isDirectory: false,
                    size: 42,
                    permissions: "rw-r--r--".to_string(),
                    lastModified: "2026-06-16T00:00:00Z".to_string(),
                }],
            }),
            error: None,
        };

        let formatted = ConversationMarkupManager::formatToolResultForMessage(&result);

        assert!(formatted.contains("Directory listing for C:\\work:"));
        assert!(formatted.contains("notes.txt"));
        assert!(!formatted.contains(r#""__type":"DirectoryListingData""#));
    }

    #[test]
    fn package_payload_keeps_nested_type_field_as_text() {
        let payload = r#"{"command":"Write-Output direct-package-ok","output":"direct-package-ok\n","exitCode":0,"sessionId":"super_admin_default_session","timedOut":false,"terminalEnvironment":{"__type":"TerminalInfoResultData","platform":"windows","terminal":"native","terminalType":"powershell","types":[]}}"#;
        let result = ToolResult {
            toolName: "super_admin:terminal".to_string(),
            success: true,
            result: ToolResultData::StringResultData(StringResultData {
                value: payload.to_string(),
            }),
            error: None,
        };

        let formatted = ConversationMarkupManager::formatToolResultForMessage(&result);

        assert!(formatted.contains(payload));
    }

    #[test]
    fn package_payload_keeps_top_level_type_field_as_text() {
        let payload = r#"{"__type":"PackageOwnedType","value":"plain package json"}"#;
        let result = ToolResult {
            toolName: "example_package:tool".to_string(),
            success: true,
            result: ToolResultData::StringResultData(StringResultData {
                value: payload.to_string(),
            }),
            error: None,
        };

        let formatted = ConversationMarkupManager::formatToolResultForMessage(&result);

        assert!(formatted.contains(payload));
    }

    #[test]
    fn package_proxy_payload_keeps_top_level_type_field_as_text() {
        let payload = r#"{"__type":"PackageOwnedType","value":"plain package json"}"#;
        let result = ToolResult {
            toolName: "package_proxy".to_string(),
            success: true,
            result: ToolResultData::StringResultData(StringResultData {
                value: payload.to_string(),
            }),
            error: None,
        };

        let formatted = ConversationMarkupManager::formatToolResultForMessage(&result);

        assert!(formatted.contains(payload));
    }

    #[test]
    fn literal_content_delimiters_survive_storage_and_send_projection() {
        use operit_model::MessagePartCodec::MessagePartCodec;
        let payload = "before </content><content>middle &amp; 🦀</content> after".to_string();
        let result = ToolResult {
            toolName: "read_file".into(),
            success: true,
            result: ToolResultData::StringResultData(StringResultData {
                value: payload.clone(),
            }),
            error: None,
        };
        assert_eq!(ConversationMarkupManager::toolResultPayload(&result), payload);
        let message = ConversationMarkupManager::formatToolResultForMessage(&result);
        let parts = MessagePartCodec::parseAssistantMarkup(&message).unwrap();
        let part = parts
            .iter()
            .find(|part| {
                part.kind == operit_model::MessagePart::MessagePartKind::ToolResult
            })
            .unwrap();
        assert_eq!(part.content, payload);
        let round_trip = MessagePartCodec::assistantMarkup(&parts);
        assert_eq!(
            MessagePartCodec::parseAssistantMarkup(&round_trip).unwrap(),
            parts
        );
    }

    /// Persistence keeps every complete result while send views alone apply previews.
    #[test]
    fn keeps_full_results_and_projects_only_the_send_view() {
        let hash = "c".repeat(64);
        let payload = "证🦀".repeat(50_000);
        let results = [ToolResult {
            toolName: "read_file".into(),
            success: true,
            result: ToolResultData::StringResultData(StringResultData {
                value: payload.clone(),
            }),
            error: None,
        }];
        let hashes = [Some(hash.clone())];
        let message =
            ConversationMarkupManager::buildToolResultMessageWithBlobHashes(&results, &hashes);
        let parts =
            operit_model::MessagePartCodec::MessagePartCodec::parseAssistantMarkup(&message)
                .unwrap();
        let part = parts
            .iter()
            .find(|part| {
                part.kind == operit_model::MessagePart::MessagePartKind::ToolResult
            })
            .expect("tool result part");
        assert_eq!(part.content, payload);
        assert_eq!(part.attributes.get("blob_hash"), Some(&hash));
        let projected = ConversationMarkupManager::projectToolResultMessageForSend(
            &message,
            &hashes,
            Some(4000),
        )
        .unwrap();
        assert!(projected.chars().count() < message.chars().count());
        assert!(projected.contains("read_tool_result"));
        assert!(!projected.contains("blob_hash="));
        assert_eq!(
            ConversationMarkupManager::projectToolResultMessageForSend(&message, &hashes, Some(0))
                .unwrap(),
            message
        );
        assert_eq!(
            ConversationMarkupManager::projectToolResultMessageForSend(&message, &hashes, None)
                .unwrap(),
            message
        );
    }

    #[test]
    fn full_error_result_keeps_its_envelope_and_complete_detail() {
        let detail = "错误🦀".repeat(50_000);
        let result = ToolResult {
            toolName: "失败工具".into(),
            success: false,
            result: ToolResultData::StringResultData(StringResultData {
                value: detail.clone(),
            }),
            error: Some("execution failed".into()),
        };
        let message = ConversationMarkupManager::formatToolResultForMessage(&result);
        let parts =
            operit_model::MessagePartCodec::MessagePartCodec::parseAssistantMarkup(&message)
                .unwrap();
        let part = parts
            .iter()
            .find(|part| {
                part.kind == operit_model::MessagePart::MessagePartKind::ToolResult
            })
            .expect("tool result part");
        assert_eq!(
            part.content,
            format!("<error>execution failed\n\n{detail}</error>")
        );
    }

    #[test]
    fn empty_batch_has_no_result_message() {
        assert_eq!(ConversationMarkupManager::buildToolResultMessage(&[]), "");
    }
}
