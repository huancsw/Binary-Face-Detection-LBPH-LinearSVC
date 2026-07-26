const query = (selector, parent = document) => parent.querySelector(selector);
const queries = (selector, parent = document) => [...parent.querySelectorAll(selector)];

function setStatus(element, message, isError = false) {
  element.textContent = message;
  element.classList.toggle("error", isError);
}

function setButtonLoading(button, loading, label) {
  button.disabled = loading;
  button.querySelector("span").textContent = loading ? label : button.dataset.label;
}

queries(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    queries(".tab").forEach((item) => item.classList.toggle("active", item === tab));
    queries(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === tab.dataset.tab));
  });
});

const threshold = query("#threshold");
threshold.addEventListener("input", () => {
  query("#threshold-value").textContent = `${Math.round(Number(threshold.value) * 100)}%`;
});

const imageInput = query("#image-input");
const uploadZone = query("#upload-zone");
const preview = query("#input-preview");
const previewCanvas = query("#preview-canvas");

function showSelectedImage(file) {
  if (!file) return;
  query("#file-name").textContent = file.name;
  query("#preview-badge").textContent = `${Math.round(file.size / 1024)} KB`;
  preview.src = URL.createObjectURL(file);
  previewCanvas.classList.add("loaded");
}

imageInput.addEventListener("change", () => showSelectedImage(imageInput.files[0]));
["dragenter", "dragover"].forEach((eventName) => uploadZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  uploadZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => uploadZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  uploadZone.classList.remove("dragging");
}));
uploadZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (!file || !file.type.startsWith("image/")) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  showSelectedImage(file);
});

query("#detect-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = query("#detect-button");
  const status = query("#detect-status");
  button.dataset.label ||= button.querySelector("span").textContent;
  setButtonLoading(button, true, "Đang phân tích ảnh...");
  setStatus(status, "Đang chạy Haar Cascade, LBPH và SVM.");
  try {
    const response = await fetch("/api/detect", { method: "POST", body: new FormData(event.currentTarget) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Không thể xử lý ảnh.");
    Object.entries(payload.stages).forEach(([name, source]) => {
      const card = query(`[data-stage="${name}"]`);
      const image = query("img", card);
      image.src = source;
      query(".stage-image", card).classList.add("loaded");
    });
    query("#result-summary").textContent = `Phát hiện ${payload.count} khuôn mặt`;
    setStatus(status, `Hoàn tất. Đã phát hiện ${payload.count} khuôn mặt.`);
  } catch (error) {
    setStatus(status, error.message, true);
  } finally {
    setButtonLoading(button, false);
  }
});

let pollingTimer = null;
async function pollTraining() {
  try {
    const response = await fetch("/api/training");
    const payload = await response.json();
    if (payload.log) {
      const log = query("#training-log");
      if (log.textContent === "Chờ khởi tạo phiên train...") log.textContent = "";
      log.textContent += payload.log;
      log.scrollTop = log.scrollHeight;
    }
    query("#log-led").classList.toggle("running", payload.running);
    if (!payload.running && pollingTimer) {
      clearInterval(pollingTimer);
      pollingTimer = null;
      setButtonLoading(query("#train-button"), false);
      setStatus(query("#train-status"), "Tiến trình train đã kết thúc.");
    }
  } catch (error) {
    setStatus(query("#train-status"), "Không thể lấy log train.", true);
  }
}

query("#train-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = query("#train-button");
  button.dataset.label ||= button.querySelector("span").textContent;
  const payload = {
    dataset: query("#dataset").value,
    output: query("#output-model").value,
    max_train: query("#max-train").value,
    max_val: query("#max-val").value,
    negatives: query("#negatives").value,
    samples: query("#samples").value,
  };
  try {
    const response = await fetch("/api/train", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Không thể bắt đầu train.");
    query("#training-log").textContent = "Khởi tạo tiến trình train...\n";
    setButtonLoading(button, true, "Đang train...");
    setStatus(query("#train-status"), "Đang train, log được cập nhật trực tiếp.");
    await pollTraining();
    pollingTimer = setInterval(pollTraining, 900);
  } catch (error) {
    setStatus(query("#train-status"), error.message, true);
  }
});

fetch("/api/health")
  .then((response) => response.json())
  .then((health) => {
    const status = query("#connection-status");
    status.classList.add("ready");
    status.lastChild.textContent = health.model_exists ? " Hệ thống sẵn sàng" : " Cần train model";
  })
  .catch(() => { query("#connection-status").lastChild.textContent = " Backend chưa sẵn sàng"; });

window.addEventListener("DOMContentLoaded", () => window.lucide?.createIcons());