/**
 * Dashboard JavaScript - Handles dashboard functionality and data loading
 */

// Dashboard data loading
// تحسين الأداء باستخدام التخزين المؤقت
const cache = new Map();
const CACHE_DURATION = 60000; // 60 ثانية

async function loadDashboardData() {
    const cacheKey = 'dashboard_data';
    const cachedData = cache.get(cacheKey);
    
    if (cachedData && (Date.now() - cachedData.timestamp < CACHE_DURATION)) {
        updateDashboardStats(cachedData.data);
        return;
    }

    try {
        const response = await fetch('/api/dashboard_data', {
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache'
            }
        });
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateDashboardStats(data.data);
            } else {
                console.error('Error loading dashboard data:', data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Update dashboard stats with loaded data
function updateDashboardStats(data) {
    // Update count cards
    document.getElementById('active-signals-count').textContent = data.active_signals || 0;
    document.getElementById('success-rate').textContent = (data.success_rate || 0) + '%';
    document.getElementById('monitored-pairs').textContent = data.monitored_pairs || 0;
    document.getElementById('active-alerts-count').textContent = data.active_alerts || 0;
}

// Load latest signals for the dashboard
function loadLatestSignals(marketType = 'all') {
    fetch(`/api/latest_signals?market_type=${marketType}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateSignalsTable(data.signals);
            } else {
                console.error('Error loading latest signals:', data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Update signals table with loaded data
function updateSignalsTable(signals) {
    const tableBody = document.getElementById('signals-table-body');
    
    // Clear table
    tableBody.innerHTML = '';
    
    if (!signals || signals.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="7" class="text-center">No signals available</td></tr>';
        return;
    }
    
    // Add signals to table
    signals.forEach(signal => {
        const row = document.createElement('tr');
        
        // Format timestamp
        const timestamp = new Date(signal.timestamp);
        const formattedTime = timestamp.toLocaleString();
        
        // Format confidence as percentage
        const confidencePercent = Math.round(signal.confidence * 100) + '%';
        
        // Set signal type class
        const signalTypeClass = (signal.signal_type === 'BUY' || signal.signal_type === 'CALL') ? 
                               'text-success' : 'text-danger';
        
        row.innerHTML = `
            <td>${signal.currency_pair}</td>
            <td class="${signalTypeClass}">${signal.signal_type}</td>
            <td>${signal.entry_price.toFixed(4)}</td>
            <td class="text-success">${signal.take_profit.toFixed(4)}</td>
            <td class="text-danger">${signal.stop_loss.toFixed(4)}</td>
            <td><div class="progress" style="height: 10px;">
                <div class="progress-bar ${signal.confidence >= 0.75 ? 'bg-success' : 'bg-warning'}" 
                     role="progressbar" 
                     style="width: ${confidencePercent};" 
                     aria-valuenow="${signal.confidence * 100}" 
                     aria-valuemin="0" 
                     aria-valuemax="100"></div>
            </div>
            <small>${confidencePercent}</small></td>
            <td>${formattedTime}</td>
        `;
        
        tableBody.appendChild(row);
    });
}

// Load watchlist currencies
function loadWatchlist() {
    fetch('/api/watchlist')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateWatchlist(data.currencies);
            } else {
                console.error('Error loading watchlist:', data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Update watchlist with loaded data
function updateWatchlist(currencies) {
    const watchlistBody = document.getElementById('watchlist-body');
    
    // Clear watchlist
    watchlistBody.innerHTML = '';
    
    if (!currencies || currencies.length === 0) {
        watchlistBody.innerHTML = '<tr><td colspan="3" class="text-center">No currencies available</td></tr>';
        return;
    }
    
    // Add currencies to watchlist
    currencies.forEach(currency => {
        const row = document.createElement('tr');
        
        // Format change with color and arrow
        const changeClass = currency.change >= 0 ? 'text-success' : 'text-danger';
        const changeArrow = currency.change >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
        const changeFormatted = Math.abs(currency.change).toFixed(2) + '%';
        
        row.innerHTML = `
            <td>${currency.symbol}</td>
            <td>${currency.price}</td>
            <td class="${changeClass}">
                <i class="fas ${changeArrow} me-1"></i>${changeFormatted}
            </td>
        `;
        
        watchlistBody.appendChild(row);
    });
}

// Load recent alerts
function loadRecentAlerts() {
    fetch('/api/recent_alerts')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateAlertsList(data.alerts);
            } else {
                console.error('Error loading recent alerts:', data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Update alerts list with loaded data
function updateAlertsList(alerts) {
    const alertsList = document.getElementById('alerts-list');
    
    // Clear list
    alertsList.innerHTML = '';
    
    if (!alerts || alerts.length === 0) {
        alertsList.innerHTML = '<li class="list-group-item text-center">No alerts available</li>';
        return;
    }
    
    // Add alerts to list
    alerts.forEach(alert => {
        const listItem = document.createElement('li');
        listItem.className = 'list-group-item';
        
        // Format timestamp
        const timestamp = new Date(alert.created_at);
        const formattedTime = timestamp.toLocaleString();
        
        // Create alert icon based on type
        let alertIcon = 'fa-bell';
        if (alert.alert_type === 'price') {
            alertIcon = 'fa-dollar-sign';
        } else if (alert.alert_type === 'pattern') {
            alertIcon = 'fa-chart-line';
        } else if (alert.alert_type === 'support_resistance') {
            alertIcon = 'fa-arrows-alt-v';
        }
        
        // Format alert message
        listItem.innerHTML = `
            <div class="d-flex w-100 justify-content-between">
                <h6 class="mb-1">
                    <i class="fas ${alertIcon} me-2"></i>${alert.currency_symbol}
                </h6>
                <small class="text-muted">${formattedTime}</small>
            </div>
            <p class="mb-1">${alert.message}</p>
            ${alert.price_level ? `<small class="text-muted">Price Level: ${alert.price_level}</small>` : ''}
        `;
        
        alertsList.appendChild(listItem);
    });
}

// Initialize performance chart
function initPerformanceChart() {
    const ctx = document.getElementById('performance-chart');
    
    if (!ctx) return;
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [], // Will be filled with dates
            datasets: [
                {
                    label: 'Success Rate (%)',
                    data: [], // Will be filled with success rate data
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Signal Count',
                    data: [], // Will be filled with signal count data
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Success Rate (%)'
                    },
                    suggestedMax: 100
                },
                y1: {
                    beginAtZero: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Signal Count'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            }
        }
    });
    
    // Load performance data
    loadPerformanceData('30d', chart);
    
    // Set up period buttons
    document.querySelectorAll('.btn-group button[data-period]').forEach(button => {
        button.addEventListener('click', function() {
            // Update active button
            document.querySelectorAll('.btn-group button[data-period]').forEach(btn => {
                btn.classList.remove('active');
            });
            this.classList.add('active');
            
            // Load data for selected period
            const period = this.getAttribute('data-period');
            loadPerformanceData(period, chart);
        });
    });
}

// Load performance data for the chart
function loadPerformanceData(period, chart) {
    fetch(`/api/performance_data?period=${period}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updatePerformanceChart(chart, data.performance);
            } else {
                console.error('Error loading performance data:', data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Update performance chart with loaded data
function updatePerformanceChart(chart, performance) {
    chart.data.labels = performance.dates;
    chart.data.datasets[0].data = performance.success_rates;
    chart.data.datasets[1].data = performance.signal_counts;
    chart.update();
}

// تهيئة نظام التداول المحسن
async function initEnhancedTradingSystem() {
    const results = await executeAllStrategies();
    updateConfirmationPanel(results);
    updateStrategySummary(results);
}

// تنفيذ جميع الاستراتيجيات
async function executeAllStrategies() {
    try {
        const response = await fetch('/api/execute_all_strategies', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        return await response.json();
    } catch (error) {
        console.error('Error executing strategies:', error);
        return null;
    }
}

// تحديث لوحة التأكيد
function updateConfirmationPanel(results) {
    if (!results) return;
    
    const panel = document.querySelector('.confirmation-status');
    const progressBar = document.querySelector('.progress-bar');
    
    // تحديث حالة الإشارة
    const signalStatus = panel.querySelector('.signal-status');
    const confidence = results.overall_confidence * 100;
    
    if (confidence >= 85) {
        panel.classList.add('bg-success-light');
        signalStatus.textContent = 'إشارة مؤكدة عالية الدقة';
        signalStatus.classList.add('text-success');
    } else if (confidence >= 70) {
        panel.classList.add('bg-warning-light');
        signalStatus.textContent = 'إشارة محتملة';
        signalStatus.classList.add('text-warning');
    } else {
        panel.classList.add('bg-danger-light');
        signalStatus.textContent = 'إشارة ضعيفة';
        signalStatus.classList.add('text-danger');
    }
    
    // تحديث مقياس الثقة
    progressBar.style.width = `${confidence}%`;
    progressBar.classList.add(confidence >= 85 ? 'bg-success' : confidence >= 70 ? 'bg-warning' : 'bg-danger');
}

// تحديث ملخص الاستراتيجيات
function updateStrategySummary(results) {
    if (!results) return;
    
    const summaryList = document.getElementById('strategy-results');
    summaryList.innerHTML = '';
    
    results.strategies.forEach(strategy => {
        const li = document.createElement('li');
        li.className = 'list-group-item d-flex justify-content-between align-items-center';
        
        li.innerHTML = `
            <span>${strategy.name}</span>
            <span class="badge ${strategy.signal ? 'bg-success' : 'bg-danger'} rounded-pill">
                ${strategy.confidence}%
            </span>
        `;
        
        summaryList.appendChild(li);
    });
}

// Initialize on document ready
document.addEventListener('DOMContentLoaded', function() {
    // Check if on dashboard page
    const performanceChart = document.getElementById('performance-chart');
    if (performanceChart) {
        initPerformanceChart();
        initEnhancedTradingSystem();
    }
});
function initAdvancedSignalChart(containerId, data) {
    const ctx = document.getElementById(containerId);
    
    if (!ctx) return;
    
    const chart = new Chart(ctx, {
        type: 'candlestick',
        data: {
            datasets: [{
                label: 'Price',
                data: data.map(d => ({
                    t: new Date(d.timestamp),
                    o: d.open,
                    h: d.high,
                    l: d.low,
                    c: d.close
                }))
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Advanced Signal Analysis'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day'
                    }
                }
            }
        }
    });
    
    return chart;
}
