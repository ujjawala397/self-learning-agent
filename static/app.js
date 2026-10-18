const form = document.querySelector("#chat-form");
const input = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");

function addMessage(role, text = "") {
  const row = document.createElement("article");
  row.className = `message ${role}`;
  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "L";
    row.append(avatar);
  }
  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = text;
  row.append(content);
  messages.append(row);
  row.scrollIntoView({ behavior: "smooth", block: "end" });
  return { row, content };
}

async function sendFeedback(payload, status, button) {
  button.disabled = true;
  status.textContent = "Saving…";
  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Feedback could not be saved.");
    status.textContent = "Feedback saved";
    return true;
  } catch (error) {
    status.textContent = error.message;
    button.disabled = false;
    return false;
  }
}

function attachFeedback(row, question, answer) {
  const controls = document.createElement("div");
  controls.className = "feedback-controls";
  const good = document.createElement("button");
  good.type = "button";
  good.className = "feedback-button";
  good.textContent = "Helpful";
  const bad = document.createElement("button");
  bad.type = "button";
  bad.className = "feedback-button";
  bad.textContent = "Not helpful";
  const status = document.createElement("span");
  status.className = "feedback-note";
  controls.append(good, bad, status);
  row.append(controls);

  good.addEventListener("click", () => {
    void sendFeedback({ question, answer, rating: "good" }, status, good);
    good.disabled = true;
    bad.disabled = true;
  });

  bad.addEventListener("click", () => {
    bad.disabled = true;
    const feedbackForm = document.createElement("form");
    feedbackForm.className = "feedback-form";
    feedbackForm.innerHTML = `
      <label>What was wrong? (optional)<textarea name="explanation" maxlength="5000"></textarea></label>
      <label>What should the answer say instead? (optional)<textarea name="correction" maxlength="10000"></textarea></label>
      <label class="confirm-row"><input type="checkbox" name="confirmed"> I confirm the explanation or correction is accurate. Confirmed details may be used as lessons for related questions.</label>
      <button class="feedback-submit" type="submit">Save feedback</button>
    `;
    feedbackForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const data = new FormData(feedbackForm);
      const explanation = data.get("explanation").trim();
      const correction = data.get("correction").trim();
      const confirmed = data.has("confirmed");
      const saved = await sendFeedback(
        { question, answer, rating: "bad", explanation, correction, confirmed },
        status,
        feedbackForm.querySelector("button"),
      );
      if (saved) {
        feedbackForm.remove();
        bad.disabled = true;
      }
    });
    row.append(feedbackForm);
    feedbackForm.querySelector("textarea").focus();
  });
}

function attachSources(row, sources) {
  if (!sources.length) return;
  const section = document.createElement("section");
  section.className = "source-list";
  const heading = document.createElement("h3");
  heading.textContent = "Report sources";
  const list = document.createElement("ul");
  for (const source of sources) {
    const item = document.createElement("li");
    item.textContent = `[${source.id}] ${source.source_file} · ${source.financial_year} · PDF page ${source.page_number ?? "unknown"}`;
    list.append(item);
  }
  section.append(heading, list);
  row.append(section);
}

function handleEvent(rawEvent, assistant, state) {
  const dataLine = rawEvent.split("\n").find((line) => line.startsWith("data: "));
  if (!dataLine) return;
  const data = JSON.parse(dataLine.slice(6));
  if (data.type === "sources") state.sources = data.sources || [];
  if (data.type === "token" && data.token) {
    state.answer += data.token;
    assistant.textContent += data.token;
    assistant.parentElement.scrollIntoView({ behavior: "smooth", block: "end" });
  }
  if (data.error) throw new Error(data.error);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || sendButton.disabled) return;

  welcome.hidden = true;
  addMessage("user", question);
  const { row, content } = addMessage("assistant");
  const state = { answer: "", sources: [] };
  input.value = "";
  input.style.height = "auto";
  sendButton.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.error || "The message could not be sent.");
    }
    if (!response.body) throw new Error("Streaming is not available in this browser.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const rawEvent of events) handleEvent(rawEvent, content, state);
      if (done) break;
    }
    if (buffer.trim()) handleEvent(buffer, content, state);
    if (!state.answer) state.answer = content.textContent;
    attachSources(row, state.sources);
    attachFeedback(row, question, state.answer);
  } catch (error) {
    const notice = document.createElement("div");
    notice.className = "error-message";
    notice.textContent = state.answer
      ? `${error.message} (partial response shown above)`
      : error.message;
    row.append(notice);
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.textContent;
    input.focus();
    input.dispatchEvent(new Event("input"));
  });
});
