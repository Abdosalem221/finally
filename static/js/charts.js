// تهيئة الرسوم البيانية
function initializeCharts() {
    // تهيئة رسم بياني للأسعار
    const priceCtx = document.getElementById('priceChart')?.getContext('2d');
    if (priceCtx) {
        new Chart(priceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'سعر السوق',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }

    // تهيئة رسم بياني للإشارات
    const signalsCtx = document.getElementById('signalsChart')?.getContext('2d');
    if (signalsCtx) {
        new Chart(signalsCtx, {
            type: 'bar',
            data: {
                labels: ['شراء', 'بيع'],
                datasets: [{
                    label: 'إشارات اليوم',
                    data: [0, 0],
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(255, 99, 132, 0.2)'
                    ],
                    borderColor: [
                        'rgb(75, 192, 192)',
                        'rgb(255, 99, 132)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }
    createPerformanceChart();
    createSignalsDistributionChart();
    createSuccessRateChart();
}

// إنشاء رسم بياني للأداء
function createPerformanceChart() {
    const ctx = document.getElementById('performance-chart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'معدل النجاح',
                data: [],
                borderColor: '#4CAF50',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'أداء النظام'
                }
            }
        }
    });
}

// إنشاء رسم بياني لتوزيع الإشارات
function createSignalsDistributionChart() {
    const ctx = document.getElementById('signals-distribution-chart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['شراء', 'بيع'],
            datasets: [{
                data: [0, 0],
                backgroundColor: ['#4CAF50', '#f44336']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                }
            }
        }
    });
}

// إنشاء رسم بياني لمعدل النجاح
function createSuccessRateChart() {
    const ctx = document.getElementById('success-rate-chart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'معدل النجاح',
                data: [],
                backgroundColor: '#2196F3'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// تحديث البيانات في الرسوم البيانية
function updateCharts(data) {
    const charts = Chart.getChart('priceChart');
    if (charts && data.prices) {
        charts.data.labels = data.prices.map(p => p.time);
        charts.data.datasets[0].data = data.prices.map(p => p.value);
        charts.update();
    }
    updatePerformanceChart(data.performance);
    updateSignalsDistributionChart(data.distribution);
    updateSuccessRateChart(data.success_rates);
}

// تحديث رسم بياني الأداء
function updatePerformanceChart(data) {
    const chart = Chart.getChart('performance-chart');
    if (!chart) return;

    chart.data.labels = data.dates;
    chart.data.datasets[0].data = data.rates;
    chart.update();
}

// تحديث رسم بياني توزيع الإشارات
function updateSignalsDistributionChart(data) {
    const chart = Chart.getChart('signals-distribution-chart');
    if (!chart) return;

    chart.data.datasets[0].data = [data.buy, data.sell];
    chart.update();
}

// تحديث رسم بياني معدل النجاح
function updateSuccessRateChart(data) {
    const chart = Chart.getChart('success-rate-chart');
    if (!chart) return;

    chart.data.labels = data.periods;
    chart.data.datasets[0].data = data.rates;
    chart.update();
}