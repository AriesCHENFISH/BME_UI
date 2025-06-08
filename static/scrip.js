let currentBmodeFile = null;  // 用于记录当前上传的图像
let ceusFileList = [];
// 自动加载最近5位患者
function loadRecentPatients() {
  fetch("/api/recent_patients")
    .then(res => res.json())
    .then(data => {
      const container = document.getElementById("recentPatients");
      container.innerHTML = "";

      data.forEach(p => {
        const div = document.createElement("div");
        div.style.display = "flex";
        div.style.justifyContent = "space-between";
        div.style.alignItems = "center";
        div.style.border = "1px solid #ccc";
        div.style.borderRadius = "8px";
        div.style.padding = "10px 16px";

        div.innerHTML = `
          <div><strong>${p.name}</strong>（编号：${p.patient_id}）</div>
          <button onclick="usePatient('${p.patient_id}', '${p.id_card}')" style="padding: 6px 12px; background: #1133ae; color: white; border: none; border-radius: 4px; cursor: pointer;">就诊</button>
        `;

        container.appendChild(div);
      });
    });
}

// 页面加载后调用
document.addEventListener("DOMContentLoaded", loadRecentPatients);

// 点击“就诊”按钮逻辑
function usePatient(patientID, idCard) {

   // 跳转到工作台（workbench）
  document.querySelector('[data-target="workbenchs"]').click();
  // 设置输入框
  document.getElementById("patientInputID").value = patientID;
  document.getElementById("patientInputCard").value = idCard;

 
}

function submitPatient() {
  const inputs = document.querySelectorAll("#patient-info input, #patient-info select");
  const name = inputs[0].value.trim();
  const gender = inputs[1].value;
  const birthdate = inputs[2].value;
  const patientID = inputs[3].value.trim();
  const idCard = inputs[4].value.trim();
  const phone = inputs[5].value.trim();
  const email = inputs[6].value.trim();
  const file = document.getElementById("fileUpload").files[0];

  if (!name || !gender || !birthdate || !patientID || !idCard || !phone || !file) {
    alert("请填写所有必填项并上传图像！");
    return;
  }

  const formData = new FormData();
  formData.append("name", name);
  formData.append("gender", gender);
  formData.append("birthdate", birthdate);
  formData.append("patient_id", patientID);
  formData.append("id_card", idCard);
  formData.append("phone", phone);
  formData.append("email", email);
  formData.append("bmode", file);

  fetch("/api/add_patient", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(result => {
    if (result.status === "success") {
      alert("✅ 患者信息添加成功！");
    } else {
      alert("❌ 添加失败：" + result.message);
    }
  })
  .catch(err => {
    console.error(err);
    alert("❌ 提交出错！");
  });
}

function loadPatientData() {
  const patientID = document.getElementById("patientInputID").value.trim();
  const idCard = document.getElementById("patientInputCard").value.trim();

  if (!patientID || !idCard) {
    alert("请输入完整的患者编号和身份证号！");
    return;
  }

  fetch("/api/patient_info", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      patient_id: patientID,
      id_card: idCard
    })
  })
  .then(res => res.json())
  .then(result => {
    if (!result.success) {
      alert(result.message || "信息不存在/身份证号错误！");
      return;
    }

    const match = result.data;
    sessionStorage.setItem("patientID", match.id);
    sessionStorage.setItem("patientName", match.name);
    sessionStorage.setItem("patientCard", match.idcard);
    sessionStorage.setItem("gender", match.gender);
    sessionStorage.setItem("age", match.age);
    sessionStorage.setItem("phone", match.phone);
    sessionStorage.setItem("email", match.email);
    sessionStorage.removeItem("doctorAdvice");

    window.location.reload();  // 重新加载页面触发自动加载影像逻辑
  })
  .catch(error => {
    console.error("查询失败：", error);
    alert("查询失败，请稍后再试！");
  });
}


// window.onload = function () {
//   // 强制页面刷新后定位到顶部
//   window.scrollTo({ top: 0, behavior: 'auto' });
//   const patientID = sessionStorage.getItem("patientID");
//   if (patientID) {
//     const formData = new FormData();
//     formData.append("patient_id", patientID);

//     fetch("/auto_load", {
//       method: "POST",
//       body: formData,
//     })
//       .then(res => res.json())
//       .then(data => {
//         if (data.error) {
//           console.warn(data.error);
//           return;
//         }
//         // 自动调用已有 startAnalysis() 的结果展示逻辑
//         displayPreviews(data.bmode_preview, data.ceus_preview);  // 只负责展示，不做分析
//       });
//   }
// };
const patientID_total = sessionStorage.getItem("patientID");
const patientName_total = sessionStorage.getItem("patientName");
window.onload = function () {
  const patientID = sessionStorage.getItem("patientID");
  if (patientID) {
    const formData = new FormData();
    formData.append("patient_id", patientID);

    fetch("/auto_load_file", {
      method: "POST",
      body: formData,
    })
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          console.warn(data.error);
          return;
        }

        // 处理 B-mode 图像
        fetch(data.bmode_path)
          .then(res => res.blob())
          .then(blob => {
            currentBmodeFile = new File([blob], "bmode.png", { type: blob.type });
            const reader = new FileReader();
            reader.onload = function (e) {
              document.getElementById('bmodePreview').src = e.target.result;
            };
            reader.readAsDataURL(currentBmodeFile);
          });

        // 处理 CEUS 图像序列
        const ceusPromises = data.ceus_paths.map((url, index) =>
          fetch(url)
            .then(res => res.blob())
            .then(blob => new File([blob], `frame_${index.toString().padStart(4, '0')}.png`, { type: blob.type }))
        );

        Promise.all(ceusPromises).then(files => {
          ceusFileList = files;

          // 显示第一帧预览
          const reader = new FileReader();
          reader.onload = function (e) {
            document.getElementById('ceusPreview').src = e.target.result;
          };
          reader.readAsDataURL(files[0]);
        });
      });
  }
};
window.addEventListener("load", () => {
  const name = sessionStorage.getItem("patientName");
  if (name) {
    document.getElementById("patientName").textContent = name;
    document.getElementById("patientID").textContent = sessionStorage.getItem("patientID");
    document.getElementById("patientCard").textContent = sessionStorage.getItem("patientCard");
    document.getElementById("patientGender").textContent = sessionStorage.getItem("gender");
    document.getElementById("patientAge").textContent = sessionStorage.getItem("age");
    document.getElementById("patientPhone").textContent = sessionStorage.getItem("phone");
    document.getElementById("patientEmail").textContent = sessionStorage.getItem("email");
  }
});



function simulateUploadFromUrl(url, type, callback) {
  fetch(url)
    .then(res => res.blob())
    .then(blob => {
      const file = new File([blob], url.split('/').pop(), { type: blob.type });

      if (type === 'bmode') {
        currentBmodeFile = file;
        const reader = new FileReader();
        reader.onload = function (e) {
          document.getElementById('bmodePreview').src = e.target.result;
        };
        reader.readAsDataURL(file);
      }

      if (type === 'ceus') {
        ceusFileList = [file];  // 只用第一帧模拟上传
        const reader = new FileReader();
        reader.onload = function (e) {
          document.getElementById('ceusPreview').src = e.target.result;
        };
        reader.readAsDataURL(file);
      }

      callback && callback();
    });
}



// Toggle Patient Info sidebar visibility
function toggleSidebar() {
    const sidebar1 = document.getElementById("sidebar1"); // Target the Patient Info sidebar
    const button = document.querySelector(".toggle-btn"); // Button that toggles the sidebar

    // Toggle the 'open' class to control visibility
    if (sidebar1.style.transform === "translateX(0%)") {
        sidebar1.style.transform = "translateX(100%)";  // Hide the Patient Info sidebar
        button.textContent = "Show Sidebar"; // Change button text
    } else {
        sidebar1.style.transform = "translateX(0%)";  // Show the Patient Info sidebar
        button.textContent = "Hide Sidebar"; // Change button text
    }
}

// Toggle terminal visibility
function toggleTerminal() {
    const terminal = document.getElementById("terminal");
    terminal.style.display = terminal.style.display === "none" ? "block" : "none";
}

// Tab switching functionality
function showTab(id) {
    const allTabs = document.querySelectorAll(".tab-content");
    const allButtons = document.querySelectorAll(".tab");

    allTabs.forEach((tab) => (tab.style.display = "none"));
    allButtons.forEach((btn) => btn.classList.remove("active"));

    document.getElementById(id).style.display = "block";
    event.target.classList.add("active");
}



 // Update progress bar (simulate file upload)
 function updateProgressBar(progress) {
  const progressBar = document.getElementById('uploadProgress');
  const progressPercentage = document.getElementById('progressPercentage');
  progressBar.style.width = progress + '%';
  progressPercentage.textContent = progress + '%';
}

// Handle B-mode Image Upload
function handleBmodeUpload(event) {
  const file = event.target.files[0];
  if (file) {
      currentBmodeFile = file;
      // Start progress bar
      let progress = 0;
      updateProgressBar(progress);

      const reader = new FileReader();
      reader.onload = function(e) {
          const imgPreview = document.getElementById('bmodePreview');
          imgPreview.src = e.target.result;  // Set image preview to uploaded image
          
          // Simulate file upload (progress)
          const interval = setInterval(function() {
              progress += 10;
              updateProgressBar(progress);
              if (progress >= 100) {
                  clearInterval(interval);
              }
          }, 100);
      };
      reader.readAsDataURL(file);
  }
}

// Handle CEUS Video Upload

function handleCeusUpload(event) {
  const files = Array.from(event.target.files)
      .filter(file => /\.(jpe?g|png|bmp|tif)$/i.test(file.name))  // 仅保留图像文件
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));

  if (files.length === 0) {
      alert("⚠️ 所选文件夹中无图像文件");
      return;
  }

  ceusFileList = files;

  // 重置进度条
  let progress = 0;
  updateProgressBar(progress);

  // 预览第一帧
  const reader = new FileReader();
  reader.onload = function(e) {
      const imgPreview = document.getElementById('ceusPreview');
      imgPreview.src = e.target.result;

      // 模拟上传进度条
      const interval = setInterval(() => {
          progress += 10;
          updateProgressBar(progress);
          if (progress >= 100) clearInterval(interval);
      }, 80);
  };
  reader.readAsDataURL(files[0]);
}
function saveAnalysisResult(patientName, patientID, resultText, reportPath, imagePath) {
  fetch('/save_result', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      patient_name: patientName,
      patient_id: patientID,
      result: resultText,
      report_path: reportPath,
      image_path: imagePath,
      time: new Date().toLocaleString('zh-CN', { hour12: false })
    })
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === "success") {
      // alert("分析结果已保存！");
      loadHistory(patientID);  // 保存后立即刷新历史
    }
  });
}


function startAnalysis() {
  if (!currentBmodeFile || ceusFileList.length === 0) {
      alert("请上传B-mode图像和CEUS图像序列！");
      return;
  }

  const formData = new FormData();
  formData.append("bmode", currentBmodeFile);

  for (let i = 0; i < ceusFileList.length; i++) {
      formData.append("ceus[]", ceusFileList[i]);
  }

  // 重置分析进度条
  let progress = 0;
  const analysisProgressBar = document.getElementById("analysisProgress");
  analysisProgressBar.style.width = "0%";

  // 模拟进度条推进
  const interval = setInterval(() => {
      progress += 10;
      analysisProgressBar.style.width = progress + "%";
      if (progress >= 90) {
          clearInterval(interval); // 停在90%，剩下交给后端完成后立即补满
      }
  }, 200);

  fetch('/analyze_all', {
      method: 'POST',
      body: formData
  })
  .then(res => res.json())
  .then(data => {
      // 推满进度条
      analysisProgressBar.style.width = "100%";

      // 显示 B-mode 结果图
      document.getElementById('backgroundImage').src = data.bmode.mask_path;

      // 显示 CEUS 结果图
      document.getElementById('backgroundImageCEUS').src = data.ceus.mask_path;

      
      // 根据 B-mode 分类结果更新 Diagnosis
      const diagBlock = document.getElementById('diagnosis');
      const result = data.bmode.classification === 0 ? '良性' : '恶性';
      const riskLevel = result === '恶性' ? 'High' : 'Low';
      const category = result === '恶性' ? '5' : '2';

      diagBlock.innerHTML = `
          <p><strong>病灶属性:</strong>
          <td><span class="status completed" style="background-color: ${result === '恶性' ? '#941919' : '#3CB371'}; color: #ddd;">${result}</span></td></p>
          
          
          
      `;

      // 同样可更新其他 tab（B-mode / CEUS / Tumor）的内容 ↓↓↓
      document.getElementById('bmode').innerHTML = `
          <p><strong>Resolution:</strong> 224 x 224</p>
          <p><strong>Format:</strong> PNG</p>
          <p><strong>Scan Depth:</strong> 5 cm</p>
          <p><strong>Focus Zones:</strong> 2</p>
          <p><strong>Gain:</strong> 45%</p>
          <p><strong>TI (Thermal Index):</strong> 0.3</p>
          <p><strong>MI (Mechanical Index):</strong> 0.8</p>
          <p><strong>Artifacts:</strong> None Detected</p>
      `;

      document.getElementById('ceus').innerHTML = `
          <p><strong>Frames:</strong> 60</p>
          <p><strong>Duration:</strong> ~10s (estimated)</p>
          <p><strong>Peak Intensity:</strong> --</p>
          <p><strong>Arrival Time:</strong> --</p>
          <p><strong>Time to Peak:</strong> --</p>
          <p><strong>Washout Start:</strong> --</p>
          <p><strong>Average Slope:</strong> --</p>
          <p><strong>Contrast Agent:</strong> SonoVue</p>
      `;

      document.getElementById('tumor').innerHTML = `
          <p><strong>Location:</strong> Upper outer quadrant</p>
          <p><strong>Size:</strong> ~14 mm</p>
          <p><strong>Area:</strong> ~150 mm²</p>
          <p><strong>Shape:</strong> Irregular</p>
          <p><strong>Margins:</strong> Spiculated</p>
          <p><strong>Vascularity:</strong> Moderate</p>
          <p><strong>Contrast Enhancement:</strong> Heterogeneous</p>
          <p><strong>Lesion Depth:</strong> 2.1 cm</p>
      `;

      alert("分析完成！");
      saveAnalysisResult(patientName_total, patientID_total, result, patientID_total+".pdf", "path2");
      

    })
    .catch(err => {
        console.error(err);
        alert("分析失败");
        analysisProgressBar.style.width = "0%";
    });
}

function loadHistory(patientID) {
  fetch('/get_history', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ patient_id: patientID })
  })
  .then(res => res.json())
  .then(results => {
    const historyList = document.getElementById("historyList");
    historyList.innerHTML = "";

    results.forEach(record => {
      const listItem = document.createElement("li");
      listItem.classList.add("completed"); // 或根据状态选择 completed / not-completed

      listItem.innerHTML = `
        <div>
          <p><strong>患者姓名:</strong> ${record.name}</p>
          <p><strong>患者编号:</strong> ${record.patient_id}</p>
          <p><strong>分析时间:</strong> ${record.analysis_time}</p>
          <p><strong>分析结果:</strong> ${record.result}</p>
        </div>
        <a href="/static/report/${record.report_path}" target="_blank" title="查看报告">
          <i class='bx bx-file icon'></i>
        </a>
      `;
      historyList.appendChild(listItem);
    });
  });
}



document.addEventListener("DOMContentLoaded", function () {
  const patientID = sessionStorage.getItem("patientID");
  if (patientID) {
    loadHistory(patientID);
    // alert("加载历史！");
  }
});








function downloadBmodeResult() {
  const imgSrc = document.getElementById("backgroundImage").src;
  const link = document.createElement('a');
  link.href = imgSrc;
  link.download = 'bmode_mask_result.png';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function downloadCeusResult() {
  const imgSrc = document.getElementById("backgroundImageCEUS").src;
  const link = document.createElement('a');
  link.href = imgSrc;
  link.download = 'ceus_mask_result.png';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

const eventSource = new EventSource("/stream_logs");
eventSource.onmessage = function(event) {
  appendToTerminal(event.data);
};
function appendToTerminal(text) {
  const terminal = document.getElementById("terminalOutput");
  terminal.innerText += "\n" + text;
  terminal.scrollTop = terminal.scrollHeight;
}


function toggleTerminal() {
    const terminal = document.getElementById("terminal");
    terminal.style.display = terminal.style.display === "none" ? "block" : "none";
  }
  
  function handleTerminalInput(event) {
    if (event.key === "Enter") {
      const input = document.getElementById("terminalInput");
      const output = document.getElementById("terminalOutput");
      const command = input.value.trim();
      if (command) {
        output.innerText += `\n> ${command}`;
        input.value = "";
        output.scrollTop = output.scrollHeight;
  
        // Simulated responses
        const responseMap = {
          "help": "Available commands: help, status, clear, version",
          "status": "Model ready. Last diagnosis: Malignant.",
          "clear": "clear",
          "version": "Model Version: 1.4.2 • Engine: ZetaAI"
        };
        const lowerCmd = command.toLowerCase();
        if (lowerCmd in responseMap) {
          if (lowerCmd === "clear") {
            output.innerText = ">";
          } else {
            output.innerText += `\n> ${responseMap[lowerCmd]}`;
          }
        } else {
          output.innerText += "\n> Unknown command. Type 'help' for a list.";
        }
      }
    }
  }
  
 

  // For B-mode Image Mask Opacity
  // For B-mode Image Mask Opacity (controlling background opacity while keeping the mask intact)
function updateOpacity() {
    const rangeValue = document.getElementById('opacityRange').value;
    const backgroundImage = document.getElementById('backgroundImage');
    const maskImage = document.getElementById('maskImage');
    
    // Adjust the background opacity based on the range value
    const opacity = rangeValue / 100;
    
    // Apply opacity to the background image (make it darker)
    backgroundImage.style.opacity = 1 - opacity;  // Darker background as opacity increases
    maskImage.style.opacity = 1;  // Keep mask fully visible, so its opacity is 1
}


function updateOpacityCEUS() {
    const rangeValue = document.getElementById('opacityRangeCEUS').value;
    const backgroundImage = document.getElementById('backgroundImageCEUS');
    const maskImage = document.getElementById('maskImageCEUS');
    
    // Adjust the background opacity based on the range value
    const opacity = rangeValue / 100;
    
    // Apply opacity to the background image (make it darker)
    backgroundImage.style.opacity = 1 - opacity;  // Darker background as opacity increases
    maskImage.style.opacity = 1;  // Keep mask fully visible, so its opacity is 1
}