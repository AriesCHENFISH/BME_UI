console.log("✅ start.js 加载成功！");

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.querySelector("button");

  btn.addEventListener("click", async (e) => {
    createRipple(e);

    const inputs = document.querySelectorAll("input");
    const patientID = inputs[0].value.trim();
    const idCardInput = inputs[1].value.trim();

    if (!patientID || !idCardInput) {
      alert("请输入完整的患者编号和身份证号！");
      return;
    }

        // ✅ 验证 patientData 是否成功加载
    if (!window.patientData || !Array.isArray(window.patientData)) {
        alert("⚠️ 无法加载患者数据文件！");
        return;
    }

    // 动态加载 patient.js 并验证身份
    await loadPatientData();

    const match = window.patientData.find(p => p.id === patientID && p.idCard === idCardInput);
    

    if (!match) {
      alert("信息不存在/身份证号错误！");
      return;
    }

    // 正确后写入 sessionStorage
    sessionStorage.setItem("patientID", match.id);
    sessionStorage.setItem("patientName", match.name);
    sessionStorage.setItem("patientCard", match.idCard);
    sessionStorage.setItem("gender", match.gender);
    sessionStorage.setItem("birthday", match.birthday);
    sessionStorage.setItem("phone", match.phone);
    sessionStorage.setItem("email", match.email);

    setTimeout(() => {
      window.location.href = "/home";
    }, 400);
  });

  const inputs = document.querySelectorAll("input");
  inputs.forEach((input) => {
    input.addEventListener("blur", () => {
      if (input.value.trim() !== "") {
        input.classList.add("used");
      } else {
        input.classList.remove("used");
      }
    });
  });
});

function createRipple(e) {
  const button = e.currentTarget;
  const rippleContainer = button.querySelector(".ripples");
  const rippleCircle = rippleContainer.querySelector(".ripplesCircle");

  const rect = button.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  rippleCircle.style.top = `${y}px`;
  rippleCircle.style.left = `${x}px`;

  rippleContainer.classList.remove("is-active");
  void rippleContainer.offsetWidth;
  rippleContainer.classList.add("is-active");
}

// 动态加载 patient.js
function loadPatientData() {
  return new Promise((resolve) => {
    const script = document.createElement("script");
    script.src = "/static/info/patient.js";
    script.onload = resolve;
    document.body.appendChild(script);
  });
}
