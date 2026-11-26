var window = {EcoData: {getLatest: () => null}};
var localStorage = {
  getItem: function() { return null; },
  setItem: function() {}
};
var fetch = function() {
  return Promise.resolve({
    json: () => Promise.resolve({ success: false })
  });
};

load('agentic-orchestrator.js');

window.Orchestrator.processQuery('Why is my carbon footprint high?', []).then(res => {
    print("SUCCESS:");
    print(JSON.stringify(res, null, 2));
}).catch(err => {
    print("ERROR:");
    print(err.toString());
    print(err.stack);
});
