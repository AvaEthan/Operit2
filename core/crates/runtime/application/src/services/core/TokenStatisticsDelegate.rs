use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use operit_providers::chat::EnhancedAIService::{
    EnhancedAIService, EnhancedAISharedState, RequestWindowEstimate,
};
use operit_store::PreferencesDataStore::{mutableStateFlow, MutableStateFlow, StateFlow};

#[derive(Debug)]
pub enum TokenCountsWriteError {
    Superseded,
    Persistence(String),
}

#[derive(Default)]
struct TokenStatisticsCache {
    cumulativeInputTokensByChatKey: HashMap<String, i64>,
    cumulativeOutputTokensByChatKey: HashMap<String, i64>,
    lastWindowSizeByChatKey: HashMap<String, i64>,
    windowRevisionByChatKey: HashMap<String, u64>,
    perRequestTokenCountByChatKey: HashMap<String, Option<(i64, i64)>>,
    requestWindowEstimateSubscriptions:
        HashMap<usize, Weak<Mutex<EnhancedAISharedState>>>,
    activeChatId: Option<String>,
}

#[derive(Clone)]
pub struct TokenStatisticsDelegate {
    pub cumulativeInputTokens: i64,
    pub cumulativeOutputTokens: i64,
    pub currentWindowSize: i64,
    pub perRequestTokenCount: Option<(i64, i64)>,
    pub cumulativeInputTokensFlow: MutableStateFlow<i64>,
    pub cumulativeOutputTokensFlow: MutableStateFlow<i64>,
    pub currentWindowSizeFlow: MutableStateFlow<i64>,
    pub perRequestTokenCountFlow: MutableStateFlow<Option<(i64, i64)>>,
    pub lastCurrentWindowSize: i64,
    cache: Arc<Mutex<TokenStatisticsCache>>,
}

impl TokenStatisticsDelegate {
    pub fn new() -> Self {
        Self {
            cumulativeInputTokens: 0,
            cumulativeOutputTokens: 0,
            currentWindowSize: 0,
            perRequestTokenCount: None,
            cumulativeInputTokensFlow: mutableStateFlow(0),
            cumulativeOutputTokensFlow: mutableStateFlow(0),
            currentWindowSizeFlow: mutableStateFlow(0),
            perRequestTokenCountFlow: mutableStateFlow(None),
            lastCurrentWindowSize: 0,
            cache: Arc::new(Mutex::new(TokenStatisticsCache::default())),
        }
    }

    fn chatKey(chatId: Option<&String>) -> String {
        chatId
            .cloned()
            .unwrap_or_else(|| "__DEFAULT_CHAT__".to_string())
    }

    fn refreshActiveFromCache(&mut self) {
        let cache = self.cache.lock().expect("token statistics cache poisoned");
        let key = Self::chatKey(cache.activeChatId.as_ref());
        let input = cache
            .cumulativeInputTokensByChatKey
            .get(&key)
            .copied()
            .unwrap_or(0);
        let output = cache
            .cumulativeOutputTokensByChatKey
            .get(&key)
            .copied()
            .unwrap_or(0);
        let window = cache
            .lastWindowSizeByChatKey
            .get(&key)
            .copied()
            .unwrap_or(0);
        let perRequest = cache
            .perRequestTokenCountByChatKey
            .get(&key)
            .cloned()
            .flatten();
        drop(cache);
        self.cumulativeInputTokens = input;
        self.cumulativeOutputTokens = output;
        self.currentWindowSize = window;
        self.perRequestTokenCount = perRequest;
        self.cumulativeInputTokensFlow.set_value(input);
        self.cumulativeOutputTokensFlow.set_value(output);
        self.currentWindowSizeFlow.set_value(window);
        self.perRequestTokenCountFlow.set_value(perRequest);
        self.lastCurrentWindowSize = window;
    }

    #[allow(non_snake_case)]
    pub fn setupCollectors(&mut self) {}

    #[allow(non_snake_case)]
    pub fn setActiveChatId(&mut self, chatId: Option<String>) {
        self.cache
            .lock()
            .expect("token statistics cache poisoned")
            .activeChatId = chatId;
        self.refreshActiveFromCache();
    }

    #[allow(non_snake_case)]
    pub fn bindChatService(&mut self, chatId: Option<String>, service: &EnhancedAIService) {
        let serviceKey = Arc::as_ptr(&service.shared_state) as usize;
        let alreadySubscribed = self
            .cache
            .lock()
            .expect("token statistics cache poisoned")
            .requestWindowEstimateSubscriptions
            .get(&serviceKey)
            .is_some_and(|service| service.strong_count() > 0);
        if !alreadySubscribed {
            self.cache
                .lock()
                .expect("token statistics cache poisoned")
                .requestWindowEstimateSubscriptions
                .insert(serviceKey, Arc::downgrade(&service.shared_state));
        }
        self.handleRequestWindowEstimateEvent(
            chatId.clone(),
            service.getRequestWindowEstimateEvent(),
        );
        self.handlePerRequestCounts(Self::chatKey(chatId.as_ref()), service.getPerRequestTokenCounts());
        if !alreadySubscribed {
            let mut delegate = self.clone();
            let flow = service.requestWindowEstimateFlow();
            flow.subscribe(move |estimate| {
                if let Some(estimate) = estimate {
                    delegate.handleRequestWindowEstimateEvent(estimate.chatId.clone(), Some(estimate));
                }
            });
        }
    }

    #[allow(non_snakeCase)]
    fn handleRequestWindowEstimateEvent(
        &mut self,
        chatId: Option<String>,
        estimate: Option<RequestWindowEstimate>,
    ) {
        let Some(estimate) = estimate else {
            return;
        };
        if estimate.chatId != chatId {
            return;
        }
        self.handleRequestWindowEstimate(Self::chatKey(chatId.as_ref()), Some(estimate.windowSize));
    }

    #[allow(non_snake_case)]
    fn handlePerRequestCounts(&mut self, key: String, counts: Option<(i64, i64)>) {
        let mut cache = self.cache.lock().expect("token statistics cache poisoned");
        if counts.is_some() {
            cache
                .perRequestTokenCountByChatKey
                .insert(key.clone(), counts);
        } else {
            cache.perRequestTokenCountByChatKey.remove(&key);
        }
        let active = Self::chatKey(cache.activeChatId.as_ref()) == key;
        drop(cache);
        if active {
            self.perRequestTokenCount = counts;
            self.perRequestTokenCountFlow.set_value(counts);
        }
    }

    #[allow(non_snake_case)]
    fn handleRequestWindowEstimate(&mut self, key: String, windowSize: Option<i64>) {
        let Some(windowSize) = windowSize else {
            return;
        };
        let mut cache = self.cache.lock().expect("token statistics cache poisoned");
        cache
            .lastWindowSizeByChatKey
            .insert(key.clone(), windowSize);
        let revision = cache
            .windowRevisionByChatKey
            .entry(key.clone())
            .or_insert(0);
        *revision = revision
            .checked_add(1)
            .expect("context window revision must not overflow");
        let active = Self::chatKey(cache.activeChatId.as_ref()) == key;
        drop(cache);
        if active {
            self.currentWindowSize = windowSize;
            self.currentWindowSizeFlow.set_value(windowSize);
            self.lastCurrentWindowSize = windowSize;
        }
    }

    #[allow(non_snake_case)]
    pub fn resetTokenStatistics(&mut self, service: Option<&mut EnhancedAIService>) {
        let mut cache = self.cache.lock().expect("token statistics cache poisoned");
        cache.cumulativeInputTokensByChatKey.clear();
        cache.cumulativeOutputTokensByChatKey.clear();
        cache.lastWindowSizeByChatKey.clear();
        cache.windowRevisionByChatKey.clear();
        cache.perRequestTokenCountByChatKey.clear();
        drop(cache);
        self.cumulativeInputTokens = 0;
        self.cumulativeOutputTokens = 0;
        self.currentWindowSize = 0;
        self.perRequestTokenCount = None;
        self.cumulativeInputTokensFlow.set_value(0);
        self.cumulativeOutputTokensFlow.set_value(0);
        self.currentWindowSizeFlow.set_value(0);
        self.perRequestTokenCountFlow.set_value(None);
        self.lastCurrentWindowSize = 0;
        if let Some(service) = service {
            service.resetTokenCounters();
        }
    }

    #[allow(non_snake_case)]
    pub fn updateCumulativeStatistics(
        &mut self,
        chatId: Option<String>,
        serviceOverride: Option<&EnhancedAIService>,
    ) {
        let activeChatId = self
            .cache
            .lock()
            .expect("token statistics cache poisoned")
            .activeChatId
            .clone();
        let key = Self::chatKey(chatId.as_ref().or(activeChatId.as_ref()));
        let Some(service) = serviceOverride else {
            return;
        };
        let currentInputTokens = service.getCurrentInputTokenCount();
        let currentOutputTokens = service.getCurrentOutputTokenCount();
        let mut cache = self.cache.lock().expect("token statistics cache poisoned");
        let newInput = cache
            .cumulativeInputTokensByChatKey
            .get(&key)
            .copied()
            .unwrap_or(0)
            + currentInputTokens;
        let newOutput = cache
            .cumulativeOutputTokensByChatKey
            .get(&key)
            .copied()
            .unwrap_or(0)
            + currentOutputTokens;
        cache
            .cumulativeInputTokensByChatKey
            .insert(key.clone(), newInput);
        cache
            .cumulativeOutputTokensByChatKey
            .insert(key.clone(), newOutput);
        let active = Self::chatKey(cache.activeChatId.as_ref()) == key;
        drop(cache);
        if active {
            self.cumulativeInputTokens = newInput;
            self.cumulativeOutputTokens = newOutput;
            self.cumulativeInputTokensFlow.set_value(newInput);
            self.cumulativeOutputTokensFlow.set_value(newOutput);
        }
    }

    fn writeTokenCountsLocked(
        cache: &mut TokenStatisticsCache,
        key: &str,
        inputTokens: i64,
        outputTokens: i64,
        windowSize: i64,
    ) -> bool {
        cache
            .cumulativeInputTokensByChatKey
            .insert(key.to_string(), inputTokens);
        cache
            .cumulativeOutputTokensByChatKey
            .insert(key.to_string(), outputTokens);
        cache
            .lastWindowSizeByChatKey
            .insert(key.to_string(), windowSize);
        let revision = cache
            .windowRevisionByChatKey
            .entry(key.to_string())
            .or_insert(0);
        *revision = revision
            .checked_add(1)
            .expect("context window revision must not overflow");
        Self::chatKey(cache.activeChatId.as_ref()) == key
    }

    fn writeTokenCounts(
        &mut self,
        key: String,
        inputTokens: i64,
        outputTokens: i64,
        windowSize: i64,
        expectedRevision: Option<u64>,
    ) -> bool {
        let mut cache = self.cache.lock().expect("token statistics cache poisoned");
        if expectedRevision.is_some_and(|expected| {
            cache
                .windowRevisionByChatKey
                .get(&key)
                .copied()
                .unwrap_or(0)
                != expected
        }) {
            return false;
        }
        let active =
            Self::writeTokenCountsLocked(&mut cache, &key, inputTokens, outputTokens, windowSize);
        drop(cache);
        if active {
            self.cumulativeInputTokens = inputTokens;
            self.cumulativeOutputTokens = outputTokens;
            self.currentWindowSize = windowSize;
            self.cumulativeInputTokensFlow.set_value(inputTokens);
            self.cumulativeOutputTokensFlow.set_value(outputTokens);
            self.currentWindowSizeFlow.set_value(windowSize);
            self.lastCurrentWindowSize = windowSize;
        }
        true
    }

    #[allow(non_snake_case)]
    pub fn persistTokenCountsIfRevision(
        &mut self,
        chatId: Option<String>,
        windowSize: i64,
        expectedRevision: u64,
        persist: impl FnOnce() -> Result<Option<(i64, i64)>, String>,
    ) -> Result<Option<(i64, i64)>, TokenCountsWriteError> {
        let key = Self::chatKey(chatId.as_ref());
        let mut cache = self.cache.lock().expect("token statistics cache poisoned");
        if cache
            .windowRevisionByChatKey
            .get(&key)
            .copied()
            .unwrap_or(0)
            != expectedRevision
        {
            return Err(TokenCountsWriteError::Superseded);
        }
        let counts = persist().map_err(TokenCountsWriteError::Persistence)?;
        if let Some((inputTokens, outputTokens)) = counts {
            let active = Self::writeTokenCountsLocked(
                &mut cache,
                &key,
                inputTokens,
                outputTokens,
                windowSize,
            );
            drop(cache);
            if active {
                self.cumulativeInputTokens = inputTokens;
                self.cumulativeOutputTokens = outputTokens;
                self.currentWindowSize = windowSize;
                self.cumulativeInputTokensFlow.set_value(inputTokens);
                self.cumulativeOutputTokensFlow.set_value(outputTokens);
                self.currentWindowSizeFlow.set_value(windowSize);
                self.lastCurrentWindowSize = windowSize;
            }
        }
        Ok(counts)
    }

    #[allow(non_snake_case)]
    pub fn setTokenCounts(
        &mut self,
        chatId: Option<String>,
        inputTokens: i64,
        outputTokens: i64,
        windowSize: i64,
    ) {
        self.writeTokenCounts(
            Self::chatKey(chatId.as_ref()),
            inputTokens,
            outputTokens,
            windowSize,
            None,
        );
    }

    #[allow(non_snake_case)]
    pub fn setTokenCountsIfRevision(
        &mut self,
        chatId: Option<String>,
        inputTokens: i64,
        outputTokens: i64,
        windowSize: i64,
        expectedRevision: u64,
    ) -> bool {
        self.writeTokenCounts(
            Self::chatKey(chatId.as_ref()),
            inputTokens,
            outputTokens,
            windowSize,
            Some(expectedRevision),
        )
    }

    #[allow(non_snake_case)]
    pub fn getWindowRevision(&self, chatId: Option<String>) -> u64 {
        let cache = self.cache.lock().expect("token statistics cache poisoned");
        let key = Self::chatKey(chatId.as_ref().or_else(|| cache.activeChatId.as_ref()));
        cache
            .windowRevisionByChatKey
            .get(&key)
            .copied()
            .unwrap_or(0)
    }

    #[allow(non_snake_case)]
    pub fn syncPersistedTokenCounts(
        &mut self,
        chatId: Option<String>,
        inputTokens: i64,
        outputTokens: i64,
        persistedWindowSize: i64,
    ) {
        let key = Self::chatKey(chatId.as_ref());
        let mut cache = self.cache.lock().expect("token statistics cache poisoned");
        let liveInputTokens = cache
            .cumulativeInputTokensByChatKey
            .get(&key)
            .copied()
            .unwrap_or(0);
        let liveOutputTokens = cache
            .cumulativeOutputTokensByChatKey
            .get(&key)
            .copied()
            .unwrap_or(0);
        let inputTokens = inputTokens.max(liveInputTokens);
        let outputTokens = outputTokens.max(liveOutputTokens);
        cache
            .cumulativeInputTokensByChatKey
            .insert(key.clone(), inputTokens);
        cache
            .cumulativeOutputTokensByChatKey
            .insert(key.clone(), outputTokens);
        let hadWindow = cache.lastWindowSizeByChatKey.contains_key(&key);
        let persistedIsNewer = (inputTokens > liveInputTokens && outputTokens >= liveOutputTokens)
            || (outputTokens > liveOutputTokens && inputTokens >= liveInputTokens);
        if !hadWindow || persistedIsNewer {
            cache
                .lastWindowSizeByChatKey
                .insert(key.clone(), persistedWindowSize);
            let revision = cache
                .windowRevisionByChatKey
                .entry(key.clone())
                .or_insert(0);
            *revision = revision
                .checked_add(1)
                .expect("context window revision must not overflow");
        }
        let window = cache
            .lastWindowSizeByChatKey
            .get(&key)
            .copied()
            .unwrap_or(0);
        let active = Self::chatKey(cache.activeChatId.as_ref()) == key;
        drop(cache);
        if active {
            self.cumulativeInputTokens = inputTokens;
            self.cumulativeOutputTokens = outputTokens;
            self.currentWindowSize = window;
            self.cumulativeInputTokensFlow.set_value(inputTokens);
            self.cumulativeOutputTokensFlow.set_value(outputTokens);
            self.currentWindowSizeFlow.set_value(window);
            self.lastCurrentWindowSize = window;
        }
    }

    #[allow(non_snake_case)]
    pub fn getCumulativeTokenCounts(&self, chatId: Option<String>) -> (i64, i64) {
        let cache = self.cache.lock().expect("token statistics cache poisoned");
        let key = Self::chatKey(chatId.as_ref().or_else(|| cache.activeChatId.as_ref()));
        (
            cache
                .cumulativeInputTokensByChatKey
                .get(&key)
                .copied()
                .unwrap_or(0),
            cache
                .cumulativeOutputTokensByChatKey
                .get(&key)
                .copied()
                .unwrap_or(0),
        )
    }

    #[allow(non_snake_case)]
    pub fn getLastCurrentWindowSize(&self, chatId: Option<String>) -> i64 {
        let cache = self.cache.lock().expect("token statistics cache poisoned");
        let key = Self::chatKey(chatId.as_ref().or_else(|| cache.activeChatId.as_ref()));
        cache
            .lastWindowSizeByChatKey
            .get(&key)
            .copied()
            .unwrap_or(0)
    }

    #[allow(non_snake_case)]
    pub fn currentWindowSizeFlow(&self) -> StateFlow<i64> {
        self.currentWindowSizeFlow.asStateFlow()
    }

    #[allow(non_snake_case)]
    pub fn cumulativeInputTokensFlow(&self) -> StateFlow<i64> {
        self.cumulativeInputTokensFlow.asStateFlow()
    }

    #[allow(non_snake_case)]
    pub fn cumulativeOutputTokensFlow(&self) -> StateFlow<i64> {
        self.cumulativeOutputTokensFlow.asStateFlow()
    }

    #[allow(non_snake_case)]
    pub fn perRequestTokenCountFlow(&self) -> StateFlow<Option<(i64, i64)>> {
        self.perRequestTokenCountFlow.asStateFlow()
    }
}

impl Default for TokenStatisticsDelegate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloned_delegates_share_windows_and_honor_the_active_chat() {
        let mut owner = TokenStatisticsDelegate::new();
        let mut worker = owner.clone();
        owner.setActiveChatId(Some("active".into()));
        worker.handleRequestWindowEstimate("active".into(), Some(100));

        assert_eq!(owner.getLastCurrentWindowSize(Some("active".into())), 100);
        assert_eq!(owner.currentWindowSizeFlow().value(), 100);

        owner.setActiveChatId(Some("other".into()));
        worker.handleRequestWindowEstimate("active".into(), Some(200));
        assert_eq!(owner.getLastCurrentWindowSize(Some("active".into())), 200);
        assert_eq!(owner.currentWindowSizeFlow().value(), 0);
    }

    #[test]
    fn request_window_events_only_update_their_owning_chat() {
        let mut delegate = TokenStatisticsDelegate::new();
        delegate.setActiveChatId(Some("chat-a".into()));
        delegate.handleRequestWindowEstimateEvent(
            Some("chat-a".into()),
            Some(RequestWindowEstimate {
                chatId: Some("chat-b".into()),
                windowSize: 900,
            }),
        );
        assert_eq!(delegate.getLastCurrentWindowSize(Some("chat-a".into())), 0);
        assert_eq!(delegate.getLastCurrentWindowSize(Some("chat-b".into())), 0);

        delegate.handleRequestWindowEstimateEvent(
            Some("chat-a".into()),
            Some(RequestWindowEstimate {
                chatId: Some("chat-a".into()),
                windowSize: 100,
            }),
        );
        assert_eq!(delegate.getLastCurrentWindowSize(Some("chat-a".into())), 100);
        assert_eq!(delegate.getWindowRevision(Some("chat-a".into())), 1);
    }

    #[test]
    fn stale_window_refresh_cannot_replace_a_newer_send_estimate() {
        let mut owner = TokenStatisticsDelegate::new();
        let mut worker = owner.clone();
        owner.setActiveChatId(Some("chat".into()));
        owner.setTokenCounts(Some("chat".into()), 10, 20, 100);
        let revision = owner.getWindowRevision(Some("chat".into()));

        worker.handleRequestWindowEstimate("chat".into(), Some(150));
        assert!(!owner.setTokenCountsIfRevision(Some("chat".into()), 10, 20, 100, revision,));
        assert_eq!(owner.getLastCurrentWindowSize(Some("chat".into())), 150);
    }

    #[test]
    fn stale_persisted_metadata_does_not_overwrite_live_counts_or_window() {
        let mut delegate = TokenStatisticsDelegate::new();
        delegate.setActiveChatId(Some("chat".into()));
        delegate.setTokenCounts(Some("chat".into()), 50, 60, 150);
        delegate.syncPersistedTokenCounts(Some("chat".into()), 30, 40, 100);

        assert_eq!(delegate.getLastCurrentWindowSize(Some("chat".into())), 150);
        assert_eq!(
            delegate.getCumulativeTokenCounts(Some("chat".into())),
            (50, 60)
        );
    }

    #[test]
    fn newer_persisted_metadata_replaces_an_older_live_window() {
        let mut delegate = TokenStatisticsDelegate::new();
        delegate.setActiveChatId(Some("chat".into()));
        delegate.setTokenCounts(Some("chat".into()), 30, 40, 100);
        delegate.syncPersistedTokenCounts(Some("chat".into()), 50, 60, 180);

        assert_eq!(delegate.getLastCurrentWindowSize(Some("chat".into())), 180);
        assert_eq!(
            delegate.getCumulativeTokenCounts(Some("chat".into())),
            (50, 60)
        );
    }

    #[test]
    fn persisted_window_commit_checks_the_live_revision_before_writing() {
        let mut delegate = TokenStatisticsDelegate::new();
        delegate.setTokenCounts(Some("chat".into()), 10, 20, 100);
        let revision = delegate.getWindowRevision(Some("chat".into()));
        delegate.handleRequestWindowEstimate("chat".into(), Some(150));

        let result =
            delegate.persistTokenCountsIfRevision(Some("chat".into()), 100, revision, || {
                panic!("a superseded refresh must not touch persistent storage")
            });

        assert!(matches!(result, Err(TokenCountsWriteError::Superseded)));
        assert_eq!(delegate.getLastCurrentWindowSize(Some("chat".into())), 150);
    }

    #[test]
    fn persisted_window_commit_publishes_db_counts_and_the_window_together() {
        let mut delegate = TokenStatisticsDelegate::new();
        delegate.setTokenCounts(Some("chat".into()), 10, 20, 100);
        let revision = delegate.getWindowRevision(Some("chat".into()));

        let result =
            delegate.persistTokenCountsIfRevision(Some("chat".into()), 120, revision, || {
                Ok(Some((10, 20)))
            });

        assert_eq!(result.unwrap(), Some((10, 20)));
        assert_eq!(delegate.getLastCurrentWindowSize(Some("chat".into())), 120);
        assert_eq!(
            delegate.getCumulativeTokenCounts(Some("chat".into())),
            (10, 20)
        );
    }
}
