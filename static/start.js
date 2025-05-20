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

    try {
      const response = await fetch("/api/patient_info", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          patient_id: patientID,
          id_card: idCardInput
        })
      });

      const result = await response.json();

      if (!result.success) {
        alert(result.message || "信息不存在/身份证号错误！");
        return;
      }

      const match = result.data;

      // 写入 sessionStorage
      sessionStorage.setItem("patientID", match.id);
      sessionStorage.setItem("patientName", match.name);
      sessionStorage.setItem("patientCard", match.idcard);
      sessionStorage.setItem("gender", match.gender);
      sessionStorage.setItem("age", match.age);
      sessionStorage.setItem("phone", match.phone);
      sessionStorage.setItem("email", match.email);
      // const age = match.idcard;
      // alert("患者年龄是: " + age + "岁");

      setTimeout(() => {
        window.location.href = "/home";
      }, 400);

    } catch (error) {
      console.error("查询失败：", error);
      alert("查询失败，请稍后再试！");
    }
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
