// Sample data for charts
const diagnosticData = {
    labels: ['Benign', 'Malignant'],
    datasets: [{
        data: [85, 35],
        backgroundColor: ['#3C91E6', '#FD7238'],
    }]
};

const intensityData = {
    labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'],
    datasets: [{
        label: 'Peak Intensity',
        data: [70, 80, 90, 85, 95],
        backgroundColor: '#FFCE26',
        borderRadius: 4,
    }]
};

const uploadData = {
    labels: Array.from({length: 50}, (_, i) => i * 10 + 1),  // 横坐标从1到500，每10个记录一次，共50个数据
    datasets: [{
        label: 'Training Loss',
        data: [
            1.20, 1.10, 1.00, 0.92, 0.85, 0.80, 0.75, 0.72, 0.68, 0.64,
            0.61, 0.58, 0.55, 0.52, 0.50, 0.47, 0.45, 0.42, 0.40, 0.38,
            0.37, 0.35, 0.34, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26,
            0.25, 0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16,
            0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07,
            0.07, 0.06, 0.06, 0.05, 0.05
        ].map(loss => loss + (Math.random() * 0.15 - 0.075)),
        fill: false,
        backgroundColor: '#ffffff',  // 填充颜色，表示训练损失区域
        borderColor: '#00ff88',      // 边框颜色，表示训练损失的折线
        tension: 0.4,
    }]
};


const tumorSizeData = {
    labels: ['Small', 'Medium', 'Large'],
    datasets: [{
        label: 'Tumor Size Distribution',
        data: [500, 800, 200],
        backgroundColor: ['#4caf50', '#ffeb3b', '#f44336'],
    }]
};

// Diagnostic Pie Chart
const ctx1 = document.getElementById('diagnosticPieChart').getContext('2d');
new Chart(ctx1, {
    type: 'pie',
    data: diagnosticData,
});

// Peak Intensity Bar Chart
const ctx2 = document.getElementById('intensityBarChart').getContext('2d');
new Chart(ctx2, {
    type: 'bar',
    data: intensityData,
});

// Daily Upload Trend Line Chart
const ctx3 = document.getElementById('uploadTrendLineChart').getContext('2d');
new Chart(ctx3, {
    type: 'line',
    data: uploadData,
});

// Tumor Size Bar Chart
const ctx4 = document.getElementById('tumorSizeBarChart').getContext('2d');
new Chart(ctx4, {
    type: 'bar',
    data: tumorSizeData,
});

// Heatmap initialization (you can add custom data and logic)
 // Heatmap data (simulated)
 const heatmapData = {
    labels: ['A', 'B', 'C', 'D', 'E'],
    datasets: [{
        label: 'Feature Heatmap',
        data: [
            { x: 0, y: 0, v: 10 }, { x: 1, y: 0, v: 20 }, { x: 2, y: 0, v: 30 },
            { x: 0, y: 1, v: 40 }, { x: 1, y: 1, v: 50 }, { x: 2, y: 1, v: 60 },
            { x: 0, y: 2, v: 70 }, { x: 1, y: 2, v: 80 }, { x: 2, y: 2, v: 90 },
            { x: 0, y: 3, v: 100 }, { x: 1, y: 3, v: 110 }, { x: 2, y: 3, v: 120 },
            { x: 0, y: 4, v: 130 }, { x: 1, y: 4, v: 140 }, { x: 2, y: 4, v: 150 },
        ],
        backgroundColor: (context) => {
            const value = context.raw.v;
            const color = value < 50 ? 'green' : value < 100 ? 'yellow' : 'red';
            return color;
        },
        borderWidth: 1,
        borderColor: '#fff',
    }]
};

// Heatmap chart initialization
const ctx5 = document.getElementById('featureHeatmap').getContext('2d');
new Chart(ctx5, {
    type: 'matrix',
    data: heatmapData,
    options: {
        responsive: true,
        scales: {
            x: {
                type: 'category',
                labels: ['A', 'B', 'C', 'D', 'E'],
            },
            y: {
                type: 'category',
                labels: ['1', '2', '3', '4', '5'],
            }
        },
        plugins: {
            legend: {
                display: false
            }
        }
    }
});