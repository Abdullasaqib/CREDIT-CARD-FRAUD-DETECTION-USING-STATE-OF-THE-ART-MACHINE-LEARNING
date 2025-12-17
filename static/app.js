const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('fileInput');
const loader = document.getElementById('loader');
const uploadSection = document.getElementById('upload-section');
const configSection = document.getElementById('config-section');
const dashboard = document.getElementById('dashboard');

let currentFilename = "";

// Drag & Drop Events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--primary)';
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border)';
    const files = e.dataTransfer.files;
    if (files.length) handleFileUpload(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFileUpload(e.target.files[0]);
});

// Step 1: Upload & Analyze (Single Step)
function handleFileUpload(file) {
    if (file.type !== "text/csv" && !file.name.endsWith('.csv')) {
        alert("Please upload a CSV file.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show loading style
    dropZone.classList.add('hidden'); // Hide dropzone
    loader.classList.remove('hidden'); // Show loader
    uploadSection.classList.add('hidden'); // Ensure upload section is hidden
    document.getElementById('dashboard').classList.add('hidden'); // Ensure dashboard is hidden if retrying

    fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                renderDashboard(data);
            } else {
                alert('Analysis Error: ' + data.detail);
                resetUpload();
            }
        })
        .catch(err => {
            console.error(err);
            alert('Analysis Failed.');
            resetUpload();
        });
}

function resetUpload() {
    loader.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    dropZone.classList.remove('hidden');
    dropZone.style.opacity = '1';
}

// Removed configSection and start-analysis-btn listeners as we are auto-detecting now


function renderDashboard(data) {
    loader.classList.add('hidden'); // Fix: Hide loader when data is ready
    dashboard.classList.remove('hidden');
    document.getElementById('filename-display').textContent = data.filename;

    // 1. Metrics
    const metrics = data.results.metrics;
    document.getElementById('accuracy-val').textContent = (metrics.accuracy * 100).toFixed(1) + '%';
    document.getElementById('accuracy-bar').style.width = (metrics.accuracy * 100) + '%';

    document.getElementById('auc-val').textContent = metrics.auc_roc.toFixed(3);
    document.getElementById('auc-bar').style.width = (metrics.auc_roc * 100) + '%';

    document.getElementById('precision-val').textContent = metrics.precision.toFixed(3);
    document.getElementById('recall-val').textContent = metrics.recall.toFixed(3);

    // 2. Feature Importance Chart
    const ctx = document.getElementById('featureChart').getContext('2d');
    const features = data.results.feature_importance.slice(0, 10);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features.map(f => f[0]),
            datasets: [{
                label: 'Importance Score',
                data: features.map(f => f[1]),
                backgroundColor: 'rgba(0, 242, 234, 0.6)',
                borderColor: '#00f2ea',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            indexAxis: 'y',
            scales: {
                x: { grid: { color: '#232732' }, ticks: { color: '#8b9bb4' } },
                y: { grid: { display: false }, ticks: { color: '#fff' } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // 2.5 Fraud Distribution (Pie Chart)
    const pieCtx = document.getElementById('pieChart').getContext('2d');
    const pieData = data.results.pie_data;
    const entityType = data.results.entity_col || 'Entity'; // e.g. 'nameOrig' or 'Account'

    // Destroy existing chart if it exists to avoid overlap on re-run
    if (window.myPieChart) window.myPieChart.destroy();

    window.myPieChart = new Chart(pieCtx, {
        type: 'doughnut',
        data: {
            // Preface the label with the Entity Name for clarity (e.g. "Account: 774")
            labels: pieData.map(d => `${entityType}: ${d.label}`),
            datasets: [{
                data: pieData.map(d => d.value),
                backgroundColor: [
                    '#ff0055', // Secondary (Red)
                    '#00f2ea', // Primary (Cyan)
                    '#fbff00', // Yellow
                    '#b300ff', // Purple
                    '#ff6600', // Orange
                    '#8b9bb4'  // Grey/Muted
                ],
                borderColor: '#14161f',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#fff',
                        font: { family: "'JetBrains Mono', monospace", size: 11 },
                        usePointStyle: true,
                        boxWidth: 8
                    }
                },
                title: {
                    display: true,
                    text: `Top Fraudulent ${entityType}s (by Amount)`,
                    color: '#8b9bb4',
                    font: { size: 12, weight: 'normal' },
                    padding: { bottom: 10 }
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            let label = context.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed !== null) {
                                label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed);
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });

    // 3. Confusion Matrix
    const cm = metrics.confusion_matrix;
    const confusionHTML = `
        <div class="conf-cell"><span class="conf-label">True Negative</span><span class="conf-val">${cm[0][0]}</span></div>
        <div class="conf-cell"><span class="conf-label">False Positive</span><span class="conf-val text-danger">${cm[0][1]}</span></div>
        <div class="conf-cell"><span class="conf-label">False Negative</span><span class="conf-val text-danger">${cm[1][0]}</span></div>
        <div class="conf-cell"><span class="conf-label">True Positive</span><span class="conf-val" style="color:var(--primary)">${cm[1][1]}</span></div>
    `;
    document.getElementById('confusion-matrix-display').innerHTML = confusionHTML;

    // 4. Anomalies Table
    const tableHead = document.getElementById('table-header');
    const tableBody = document.getElementById('table-body');
    const anomalies = data.results.anomalies;

    if (anomalies.length > 0) {
        const keys = Object.keys(anomalies[0]);
        let headerHTML = '';
        keys.forEach(k => {
            headerHTML += `<th>${k}</th>`;
        });
        tableHead.innerHTML = headerHTML;

        let bodyHTML = '';
        anomalies.forEach(row => {
            bodyHTML += '<tr>';
            keys.forEach(k => {
                let val = row[k];
                if (typeof val === 'number' && val % 1 !== 0) val = val.toFixed(4);

                if (k === 'Predicted_Fraud_Prob') {
                    bodyHTML += `<td style="color:var(--secondary); font-weight:bold;">${val}</td>`;
                } else {
                    bodyHTML += `<td>${val}</td>`;
                }
            });
            bodyHTML += '</tr>';
        });
        tableBody.innerHTML = bodyHTML;
    }
}
