const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function settingsWith(saved) {
    const storage = new Map([['snapSettings', JSON.stringify(saved)]]);
    const context = vm.createContext({
        EventTarget,
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
