const allSideMenu = document.querySelectorAll('#sidebar .side-menu.top li a');

allSideMenu.forEach(item => {
    const li = item.parentElement;

    item.addEventListener('click', function () {
        allSideMenu.forEach(i => {
            i.parentElement.classList.remove('active');
        })
        li.classList.add('active');
    })
});

// TOGGLE SIDEBAR
const menuBar = document.querySelector('#content nav .bx.bx-menu');
const sidebar = document.getElementById('sidebar');

// Sidebar toggle işlemi
menuBar.addEventListener('click', function () {
    sidebar.classList.toggle('hide');
});

// Sayfa yüklendiğinde ve boyut değişimlerinde sidebar durumunu ayarlama
function adjustSidebar() {
    if (window.innerWidth <= 576) {
        sidebar.classList.add('hide');  // 576px ve altı için sidebar gizli
        sidebar.classList.remove('show');
    } else {
        sidebar.classList.remove('hide');  // 576px'den büyükse sidebar görünür
        sidebar.classList.add('show');
    }
}

// Sayfa yüklendiğinde ve pencere boyutu değiştiğinde sidebar durumunu ayarlama
window.addEventListener('load', adjustSidebar);
window.addEventListener('resize', adjustSidebar);

// Arama butonunu toggle etme
const searchButton = document.querySelector('#content nav form .form-input button');
const searchButtonIcon = document.querySelector('#content nav form .form-input button .bx');
const searchForm = document.querySelector('#content nav form');

searchButton.addEventListener('click', function (e) {
    if (window.innerWidth < 768) {
        e.preventDefault();
        searchForm.classList.toggle('show');
        if (searchForm.classList.contains('show')) {
            searchButtonIcon.classList.replace('bx-search', 'bx-x');
        } else {
            searchButtonIcon.classList.replace('bx-x', 'bx-search');
        }
    }
})

// Dark Mode Switch
const switchMode = document.getElementById('switch-mode');

switchMode.addEventListener('change', function () {
    if (this.checked) {
        document.body.classList.add('dark');
    } else {
        document.body.classList.remove('dark');
    }
})

// Notification Menu Toggle
document.querySelector('.notification').addEventListener('click', function () {
    document.querySelector('.notification-menu').classList.toggle('show');
    document.querySelector('.profile-menu').classList.remove('show'); // Close profile menu if open
});

// Profile Menu Toggle
document.querySelector('.profile').addEventListener('click', function () {
    document.querySelector('.profile-menu').classList.toggle('show');
    document.querySelector('.notification-menu').classList.remove('show'); // Close notification menu if open
});

// Close menus if clicked outside
window.addEventListener('click', function (e) {
    if (!e.target.closest('.notification') && !e.target.closest('.profile')) {
        document.querySelector('.notification-menu').classList.remove('show');
        document.querySelector('.profile-menu').classList.remove('show');
    }
});

// Menülerin açılıp kapanması için fonksiyon
    function toggleMenu(menuId) {
      var menu = document.getElementById(menuId);
      var allMenus = document.querySelectorAll('.menu');

      // Diğer tüm menüleri kapat
      allMenus.forEach(function(m) {
        if (m !== menu) {
          m.style.display = 'none';
        }
      });

      // Tıklanan menü varsa aç, yoksa kapat
      if (menu.style.display === 'none' || menu.style.display === '') {
        menu.style.display = 'block';
      } else {
        menu.style.display = 'none';
      }
    }

    // Başlangıçta tüm menüleri kapalı tut
    document.addEventListener("DOMContentLoaded", function() {
      var allMenus = document.querySelectorAll('.menu');
      allMenus.forEach(function(menu) {
        menu.style.display = 'none';
      });
    });

    function viewFile(caseId) {
        alert("Viewing file for case: " + caseId);
        // 这里可以跳转或打开文件详情页面，例如：
        // window.location.href = `/case/${caseId}`;
    }

    const links = document.querySelectorAll('.side-menu a');
  const sections = document.querySelectorAll('.content-section');

  links.forEach(link => {
    link.addEventListener('click', function (e) {
      e.preventDefault();

      // 1. 移除激活状态
      links.forEach(l => l.parentElement.classList.remove('active'));

      // 2. 设置当前为激活
      this.parentElement.classList.add('active');

      // 3. 隐藏所有内容块
      sections.forEach(sec => sec.style.display = 'none');

      // 4. 显示对应的内容块
      const targetId = this.getAttribute('data-target');
      const targetSection = document.getElementById(targetId);
      if (targetSection) {
        targetSection.style.display = 'block';
      }
    });
  });

  document.getElementById("downloadReportBtn").addEventListener("click", () => {
    fetch("/generate_report", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: sessionStorage.getItem("patientName"),
        age: sessionStorage.getItem("age"),
        patientID: sessionStorage.getItem("patientID"),
        patientCard: sessionStorage.getItem("patientCard"),
        result: document.querySelector("#diagnosis .status").textContent.trim(),
        bmodeMask: document.getElementById("backgroundImage").src,
        ceusMask: document.getElementById("backgroundImageCEUS").src,
        time: new Date().toLocaleString('zh-CN', { hour12: false }),
        doctorAdvice: sessionStorage.getItem("doctorAdvice") || "无",
        bmodePre: document.getElementById("bmodePreview").src,
        ceusPre: document.getElementById("ceusPreview").src,
      }),
    })
      .then((res) => res.blob())
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "diagnosis_report.jpg";
        link.click();
        URL.revokeObjectURL(url);
      })
      .catch((err) => {
        console.error("报告生成失败", err);
        alert("报告生成失败，请稍后重试！");
      });
  });
  
  document.getElementById("saveAdviceBtn").addEventListener("click", () => {
    const advice = document.getElementById("doctorAdvice").value.trim();
    sessionStorage.setItem("doctorAdvice", advice);
    alert("✅ 医生建议已保存！");
  });
  