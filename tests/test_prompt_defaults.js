const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function settingsWith(saved, fetchImpl) {
    const storage = new Map([['snapSettings', JSON.stringify(saved)]]);
    const context = vm.createContext({
        EventTarget,
        fetch: fetchImpl,
        window: {},
        localStorage: {
            getItem: key => storage.get(key) || null,
            setItem: (key, value) => storage.set(key, value),
        },
    });
    vm.runInContext(fs.readFileSync('static/js/settings.js', 'utf8') + '\nglobalThis.Settings = SettingsManager;', context);
    const settings = Object.create(context.Settings.prototype);
    settings.prompts = JSON.parse(fs.readFileSync('config/prompts.json', 'utf8'));
    settings.models = [{ id: 'qwen3.8-flash' }, { id: 'deepseek-v4.1-flash-1017' }];
    settings.restoreLocalPrefs();
    return { settings, saved: () => JSON.parse(storage.get('snapSettings')) };
}

test('existing users switch to the new default once without losing model or network preferences', () => {
    const { settings, saved } = settingsWith({
        promptId: 'ACM_hard', model: 'qwen3.8-flash', tier: 'max', proxyEnabled: true,
    });
    assert.equal(settings.currentPromptId, 'beisen_finance');
    assert.equal(saved().defaultPromptVersion, 'beisen-finance-v1');
    assert.equal(saved().model, 'qwen3.8-flash');
    assert.equal(saved().tier, 'max');
    assert.equal(saved().proxyEnabled, true);
});

test('manual prompt choices persist after the one-time default change', () => {
    const { settings } = settingsWith({ promptId: 'ACM_hard', defaultPromptVersion: 'beisen-finance-v1' });
    assert.equal(settings.currentPromptId, 'ACM_hard');
});

test('new users and missing saved prompts use the Beisen default', () => {
    assert.equal(settingsWith({}).settings.currentPromptId, 'beisen_finance');
    assert.equal(settingsWith({ promptId: 'deleted', defaultPromptVersion: 'beisen-finance-v1' }).settings.currentPromptId, 'beisen_finance');
});

test('the coding preset is selectable and sent as the system prompt', () => {
    const { settings } = settingsWith({ promptId: 'niuke_python', defaultPromptVersion: 'beisen-finance-v1' });
    assert.equal(settings.isBuiltinPrompt('niuke_python'), true);
    assert.equal(settings.currentPromptId, 'niuke_python');
    assert.equal(settings.getSettings().systemPrompt, settings.prompts.niuke_python.content);
});

test('saved Bailian DeepSeek selection migrates to the 1017 snapshot', () => {
    const { settings, saved } = settingsWith({
        model: 'deepseek-v4.1-flash', tier: 'max', promptId: 'niuke_python',
        defaultPromptVersion: 'beisen-finance-v1', proxyEnabled: true,
    });
    assert.equal(settings.currentModelId, 'deepseek-v4.1-flash-1017');
    assert.equal(saved().model, 'deepseek-v4.1-flash-1017');
    assert.equal(saved().tier, 'max');
    assert.equal(saved().promptId, 'niuke_python');
    assert.equal(saved().proxyEnabled, true);
});

test('official DeepSeek requires its own key while Bailian keeps the Alibaba key', () => {
    const { settings } = settingsWith({});
    settings.apiKeyValues = { AlibabaApiKey: 'bailian-test' };
    const official = { id: 'deepseek-flash', provider: 'deepseek' };
    const bailian = { id: 'deepseek-v4.1-flash-1017', provider: 'alibaba' };
    assert.equal(settings.keyIdOf(official), 'DeepseekApiKey');
    assert.equal(settings.hasKeyFor(official), false);
    assert.equal(settings.hasKeyFor(bailian), true);
    settings.apiKeyValues.DeepseekApiKey = 'official-test';
    assert.equal(settings.hasKeyFor(official), true);
});


test('keys update only after confirmed server persistence; failures keep the previous key', async () => {
    for (const failure of ['offline', 'http', 'application']) {
        const { settings } = settingsWith({}, async () => {
            if (failure === 'offline') throw new Error('network');
            return { ok: failure !== 'http', json: async () => ({ success: false, message: 'failed' }) };
        });
        settings.apiKeyValues = { DeepseekApiKey: 'previous' };
        let changes = 0;
        settings.emitChange = () => changes++;
        assert.equal(await settings.saveApiKey('DeepseekApiKey', 'replacement'), false);
        assert.equal(settings.apiKeyValues.DeepseekApiKey, 'previous');
        assert.equal(changes, 0);
    }
    const { settings } = settingsWith({}, async () => ({ ok: true, json: async () => ({ success: true }) }));
    settings.apiKeyValues = {};
    let changes = 0;
    settings.emitChange = () => changes++;
    assert.equal(await settings.saveApiKey('DeepseekApiKey', 'saved-test'), true);
    assert.equal(settings.apiKeyValues.DeepseekApiKey, 'saved-test');
    assert.equal(changes, 1);
});
