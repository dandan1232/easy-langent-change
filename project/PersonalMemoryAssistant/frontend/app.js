const messageList = document.querySelector("#messageList");
const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const clearButton = document.querySelector("#clearButton");
const memoryGrid = document.querySelector("#memoryGrid");
const statusDot = document.querySelector("#statusDot");

const categoryLabels = {
  preferences: "偏好",
  plans: "计划",
  constraints: "限制",
  facts: "事实",
};

function setStatus(text, busy = false) {
  statusDot.textContent = text;
  statusDot.classList.toggle("is-busy", busy);
}

function appendMessage(role, text, meta = []) {
  const article = document.createElement("article");
  article.className = `message ${role}-message`;

  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  article.appendChild(paragraph);

  if (meta.length > 0) {
    const metaList = document.createElement("div");
    metaList.className = "meta-list";
    meta.forEach((item) => {
      const pill = document.createElement("span");
      pill.className = "meta-pill";
      pill.textContent = item;
      metaList.appendChild(pill);
    });
    article.appendChild(metaList);
  }

  messageList.appendChild(article);
  messageList.scrollTop = messageList.scrollHeight;
}

function renderMemories(memoryData = {}) {
  memoryGrid.innerHTML = "";
  let hasMemory = false;

  Object.entries(categoryLabels).forEach(([category, label]) => {
    const items = memoryData[category] || [];
    if (items.length === 0) {
      return;
    }
    hasMemory = true;

    const section = document.createElement("section");
    section.className = "memory-section";

    const heading = document.createElement("h3");
    heading.textContent = label;
    section.appendChild(heading);

    const list = document.createElement("ul");
    items.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item.content;
      list.appendChild(li);
    });
    section.appendChild(list);
    memoryGrid.appendChild(section);
  });

  if (!hasMemory) {
    const empty = document.createElement("p");
    empty.className = "empty-memory";
    empty.textContent = "暂无长期记忆。";
    memoryGrid.appendChild(empty);
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "请求失败。");
  }
  return data;
}

async function loadMemories() {
  try {
    const data = await requestJson("/api/memories");
    renderMemories(data.memory_data);
  } catch (error) {
    appendMessage("error", error.message);
  }
}

async function sendMessage(message) {
  appendMessage("user", message);
  messageInput.value = "";
  sendButton.disabled = true;
  setStatus("思考中", true);

  try {
    const data = await requestJson("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    const meta = data.new_memories.map((memory) => `已记住：${memory}`);
    appendMessage("assistant", data.visible_reply, meta);
    renderMemories(data.memories.memory_data);
    setStatus("已同步");
  } catch (error) {
    appendMessage("error", error.message);
    setStatus("异常");
  } finally {
    sendButton.disabled = false;
    messageInput.focus();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    return;
  }
  sendMessage(message);
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    messageInput.value = button.dataset.example;
    messageInput.focus();
  });
});

clearButton.addEventListener("click", async () => {
  setStatus("清理中", true);
  try {
    await requestJson("/api/clear", { method: "POST" });
    renderMemories({});
    appendMessage("assistant", "长期记忆和本轮对话已经清空。");
    setStatus("就绪");
  } catch (error) {
    appendMessage("error", error.message);
    setStatus("异常");
  }
});

loadMemories();
