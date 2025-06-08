console.log("✅ start.js 加载成功！");
function loginDoctor() {
  const id = document.getElementById("doctorID").value.trim();
  const pwd = document.getElementById("doctorPwd").value.trim();

  if (!id || !pwd) {
    
    return;
  }

  if (id === "123456" && pwd === "123456") {
    window.location.href = "/home";
  } else {
    alert("医生工号或密码错误！");
  }
}

loginDoctor()
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.querySelector("button");

  btn.addEventListener("click", async (e) => {
    createRipple(e);

    const inputs = document.querySelectorAll("input");
    const patientID = inputs[0].value.trim();
    const idCardInput = inputs[1].value.trim();

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

