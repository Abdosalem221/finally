/**
 * Advanced Analytics Dashboard JavaScript
 * Handles data loading, chart initialization, and interactive filtering for the advanced analytics dashboard
 */

// Global chart objects
let signalAccuracyChart = null;
let strategyPerformanceChart = null;
let enhancementImpactChart = null;
let timeframePerformanceChart = null;
let currencyPerformanceChart = null;
let marketRegimeChart = null;
let aiModelsChart = null;

// Global settings
let selectedTimeRange = 30;
let autoRefreshEnabled = true;
let refreshInterval = 60000; // 1 minute
let selectedCurrency = 'EUR/USD';
let selectedTimeframe = '1h';

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    initializeEventListeners();
    startAutoRefresh();
    loadInitialData();
});

function initializeCharts() {
    // Initialize all chart objects with ApexCharts
    signalAccuracyChart = new ApexCharts(
        document.querySelector("#signal-accuracy-chart"), 
        getSignalAccuracyChartOptions()
    );
    signalAccuracyChart.render();

    // Initialize other charts similarly...
}

function initializeEventListeners() {
    // Currency selector
    document.querySelector('#currency-selector').addEventListener('change', function(e) {
        selectedCurrency = e.target.value;
        refreshData();
    });

    // Timeframe selector
    document.querySelector('#timeframe-selector').addEventListener('change', function(e) {
        selectedTimeframe = e.target.value;
        refreshData();
    });

    // Auto-refresh toggle
    document.querySelector('#auto-refresh-toggle').addEventListener('change', function(e) {
        autoRefreshEnabled = e.target.checked;
        if (autoRefreshEnabled) {
            startAutoRefresh();
        } else {
            stopAutoRefresh();
        }
    });

    // Refresh button
    document.querySelector('#refresh-data').addEventListener('click', function() {
        refreshData();
    });

    // Export data button
    document.querySelector('#export-data').addEventListener('click', function() {
        exportData();
    });
}

function startAutoRefresh() {
    if (window.refreshTimer) {
        clearInterval(window.refreshTimer);
    }
    window.refreshTimer = setInterval(refreshData, refreshInterval);
}

function stopAutoRefresh() {
    if (window.refreshTimer) {
        clearInterval(window.refreshTimer);
    }
}

function refreshData() {
    showLoadingSpinner();
    
    Promise.all([
        fetchSignalAccuracyData(),
        fetchStrategyPerformanceData(),
        fetchMarketRegimeData(),
        fetchAIAnalysisData()
    ])
    .then(([accuracyData, strategyData, regimeData, aiData]) => {
        updateCharts(accuracyData, strategyData, regimeData, aiData);
        hideLoadingSpinner();
    })
    .catch(error => {
        console.error('Error refreshing data:', error);
        showError('Failed to refresh data');
        hideLoadingSpinner();
    });
}

function fetchSignalAccuracyData() {
    return fetch(`/api/analytics/signal-accuracy?currency=${selectedCurrency}&timeframe=${selectedTimeframe}`)
        .then(response => response.json());
}

function updateCharts(accuracyData, strategyData, regimeData, aiData) {
    // Update signal accuracy chart
    signalAccuracyChart.updateSeries([{
        name: 'Success Rate',
        data: accuracyData.series
    }]);

    // Update other charts...
}

function showLoadingSpinner() {
    document.querySelectorAll('.chart-container').forEach(container => {
        container.classList.add('loading');
    });
}

function hideLoadingSpinner() {
    document.querySelectorAll('.chart-container').forEach(container => {
        container.classList.remove('loading');
    });
}

function showError(message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.querySelector('#alerts-container').appendChild(alertDiv);
}

function exportData() {
    const data = {
        currency: selectedCurrency,
        timeframe: selectedTimeframe,
        charts: {
            signalAccuracy: signalAccuracyChart.w.globals.series,
            strategyPerformance: strategyPerformanceChart.w.globals.series,
            marketRegime: marketRegimeChart.w.globals.series
        }
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analytics-export-${new Date().toISOString()}.json`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

function getSignalAccuracyChartOptions() {
    return {
        chart: {
            height: 350,
            type: 'line',
            animations: {
                enabled: true,
                easing: 'easeinout',
                speed: 800
            },
            toolbar: {
                show: true,
                tools: {
                    download: true,
                    selection: true,
                    zoom: true,
                    zoomin: true,
                    zoomout: true,
                    pan: true,
                    reset: true
                }
            }
        },
        stroke: {
            curve: 'smooth',
            width: 3
        },
        series: [{
            name: 'Success Rate',
            data: []
        }],
        xaxis: {
            type: 'datetime'
        },
        yaxis: {
            labels: {
                formatter: function(val) {
                    return val.toFixed(2) + "%"
                }
            }
        },
        title: {
            text: 'Signal Success Rate Over Time',
            align: 'left'
        },
        grid: {
            borderColor: '#e7e7e7',
            row: {
                colors: ['#f3f3f3', 'transparent'],
                opacity: 0.5
            }
        },
        markers: {
            size: 6
        }
    };
}
function initEventListeners() {
    // Time range selector
    document.querySelectorAll('.time-range').forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Update active class
            document.querySelectorAll('.time-range').forEach(el => {
                el.classList.remove('active');
            });
            this.classList.add('active');
            
            // Update selected range text
            document.getElementById('selectedRange').textContent = this.textContent;
            
            // Update global timerange setting
            selectedTimeRange = parseInt(this.getAttribute('data-range'));
            
            // Reload dashboard data with new range
            loadDashboardData();
        });
    });
    
    // Strategy type filter
    document.querySelectorAll('.strategy-btn').forEach(item => {
        item.addEventListener('click', function() {
            // Update active class
            document.querySelectorAll('.strategy-btn').forEach(el => {
                el.classList.remove('active');
            });
            this.classList.add('active');
            
            // Update strategy chart with filter
            let marketType = this.getAttribute('data-period');
            loadStrategyData(marketType);
        });
    });
    
    // Enhancement category filter
    document.querySelectorAll('.enhancement-btn').forEach(item => {
        item.addEventListener('click', function() {
            // Update active class
            document.querySelectorAll('.enhancement-btn').forEach(el => {
                el.classList.remove('active');
            });
            this.classList.add('active');
            
            // Update enhancement chart with filter
            let category = this.getAttribute('data-category');
            loadEnhancementData(category);
        });
    });
    
    // AI model type filter
    document.querySelectorAll('.ai-model-btn').forEach(item => {
        item.addEventListener('click', function() {
            // Update active class
            document.querySelectorAll('.ai-model-btn').forEach(el => {
                el.classList.remove('active');
            });
            this.classList.add('active');
            
            // Update AI model chart with filter
            let modelType = this.getAttribute('data-model');
            loadAiModelData(modelType);
        });
    });
    
    // Refresh dashboard button
    document.getElementById('refreshDashboard').addEventListener('click', function() {
        loadDashboardData();
    });
    
    // Refresh activity button
    document.getElementById('refreshActivity').addEventListener('click', function() {
        loadRecentActivity();
    });
    
    // Refresh verification status
    document.getElementById('refreshVerifications').addEventListener('click', function() {
        loadVerificationStatus();
    });
}

// Load all dashboard data
function loadDashboardData() {
    // Show loading spinner
    showLoadingSpinner();
    
    // Load key metrics
    loadKeyMetrics();
    
    // Load success rate history
    loadSuccessRateHistory();
    
    // Load strategy performance data
    loadStrategyData('all');
    
    // Load enhancement impact data
    loadEnhancementData('all');
    
    // Load timeframe performance data
    loadTimeframeData();
    
    // Load currency performance data
    loadCurrencyData();
    
    // Load market regime data
    loadMarketRegimeData();
    
    // Load AI model performance data
    loadAiModelData('all');
    
    // Load recent activity
    loadRecentActivity();
    
    // Load verification status
    loadVerificationStatus();
    
    // Hide loading spinner after all data is loaded
    setTimeout(hideLoadingSpinner, 1000);
}

// Show loading spinner
function showLoadingSpinner() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = 'flex';
    }
}

// Hide loading spinner
function hideLoadingSpinner() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = 'none';
    }
}

// Load key metrics data
function loadKeyMetrics() {
    // Fetch key metrics from API
    fetch(`/api/analytics/performance_metrics?days=${selectedTimeRange}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update metrics on the dashboard
                document.getElementById('overallSuccessRate').textContent = data.metrics.success_rate + '%';
                document.getElementById('avgRiskReward').textContent = data.metrics.avg_risk_reward;
                document.getElementById('signalRate').textContent = data.metrics.signal_rate + '/day';
                document.getElementById('falsePositives').textContent = data.metrics.false_positives_prevented;
            } else {
                console.error('Error loading key metrics:', data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching key metrics:', error);
        });
}

// Load success rate history
function loadSuccessRateHistory() {
    // Fetch success rate history from API
    fetch(`/api/analytics/success_rate_history?days=${selectedTimeRange}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update chart with data
                updateSignalAccuracyChart(data.labels, data.success_rates, data.verification_rates);
            } else {
                console.error('Error loading success rate history:', data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching success rate history:', error);
        });
}

// Update the signal accuracy chart with new data
function updateSignalAccuracyChart(labels, successRates, verificationRates) {
    const ctx = document.getElementById('signalAccuracyChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (signalAccuracyChart) {
        signalAccuracyChart.destroy();
    }
    
    // Create new chart
    signalAccuracyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Success Rate (%)',
                    data: successRates,
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Verification Rate (%)',
                    data: verificationRates,
                    borderColor: '#17a2b8',
                    backgroundColor: 'rgba(23, 162, 184, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 80,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.raw + '%';
                        }
                    }
                }
            }
        }
    });
}

// Load strategy performance data
function loadStrategyData(marketType) {
    // Fetch strategy data from API
    fetch(`/api/analytics/top_strategies?market_type=${marketType}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Extract data for chart
                const strategies = data.strategies.slice(0, 5); // Limit to top 5
                const labels = strategies.map(s => s.name);
                const successRates = strategies.map(s => s.success_rate);
                
                // Determine background colors based on success rate
                const backgroundColors = successRates.map(rate => {
                    if (rate >= 93) {
                        return 'rgba(40, 167, 69, 0.7)'; // Green for high success
                    } else if (rate >= 90) {
                        return 'rgba(23, 162, 184, 0.7)'; // Blue for good success
                    } else {
                        return 'rgba(255, 193, 7, 0.7)'; // Yellow for moderate success
                    }
                });
                
                const borderColors = backgroundColors.map(color => color.replace('0.7', '1'));
                
                // Update chart
                updateStrategyPerformanceChart(labels, successRates, backgroundColors, borderColors);
            } else {
                console.error('Error loading strategy data:', data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching strategy data:', error);
        });
}

// Update the strategy performance chart with new data
function updateStrategyPerformanceChart(labels, successRates, backgroundColors, borderColors) {
    const ctx = document.getElementById('strategyPerformanceChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (strategyPerformanceChart) {
        strategyPerformanceChart.destroy();
    }
    
    // Create new chart
    strategyPerformanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Success Rate (%)',
                data: successRates,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: {
                x: {
                    min: 80,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.raw + '%';
                        }
                    }
                }
            }
        }
    });
}

// Load enhancement impact data
function loadEnhancementData(category) {
    // Fetch enhancement data from API
    fetch(`/api/analytics/enhancement_impact?category=${category}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Extract data for chart
                const enhancements = data.enhancements.slice(0, 5); // Limit to top 5
                const labels = enhancements.map(e => e.name);
                const impactScores = enhancements.map(e => e.impact_score);
                
                // Update chart
                updateEnhancementImpactChart(labels, impactScores);
            } else {
                console.error('Error loading enhancement data:', data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching enhancement data:', error);
        });
}

// Update the enhancement impact chart with new data
function updateEnhancementImpactChart(labels, impactScores) {
    const ctx = document.getElementById('enhancementImpactChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (enhancementImpactChart) {
        enhancementImpactChart.destroy();
    }
    
    // Create new chart
    enhancementImpactChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Impact Score',
                data: impactScores,
                backgroundColor: [
                    'rgba(111, 66, 193, 0.7)',
                    'rgba(111, 66, 193, 0.7)',
                    'rgba(111, 66, 193, 0.7)',
                    'rgba(111, 66, 193, 0.7)',
                    'rgba(111, 66, 193, 0.7)'
                ],
                borderColor: [
                    'rgba(111, 66, 193, 1)',
                    'rgba(111, 66, 193, 1)',
                    'rgba(111, 66, 193, 1)',
                    'rgba(111, 66, 193, 1)',
                    'rgba(111, 66, 193, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: {
                x: {
                    min: 0,
                    max: 10
                }
            }
        }
    });
}

// Load timeframe performance data
function loadTimeframeData() {
    // Fetch timeframe data from API
    fetch('/api/analytics/timeframe_performance')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Extract data for chart
                const timeframes = data.timeframe_performance;
                const labels = timeframes.map(t => t.timeframe);
                const successRates = timeframes.map(t => t.success_rate);
                
                // Update chart
                updateTimeframePerformanceChart(labels, successRates);
            } else {
                console.error('Error loading timeframe data:', data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching timeframe data:', error);
        });
}

// Update the timeframe performance chart with new data
function updateTimeframePerformanceChart(labels, successRates) {
    const ctx = document.getElementById('timeframePerformanceChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (timeframePerformanceChart) {
        timeframePerformanceChart.destroy();
    }
    
    // Create new chart
    timeframePerformanceChart = new Chart(ctx, {
        type: 'polarArea',
        data: {
            labels: labels,
            datasets: [{
                label: 'Success Rate',
                data: successRates,
                backgroundColor: [
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(255, 206, 86, 0.7)',
                    'rgba(75, 192, 192, 0.7)',
                    'rgba(153, 102, 255, 0.7)',
                    'rgba(255, 159, 64, 0.7)',
                    'rgba(199, 199, 199, 0.7)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    min: 80,
                    max: 100
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.raw + '% success rate';
                        }
                    }
                }
            }
        }
    });
}

// Load currency performance data
function loadCurrencyData() {
    // Fetch currency data from API
    fetch('/api/analytics/currency_performance')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Extract data for chart
                const currencies = data.currency_performance;
                const labels = currencies.map(c => c.currency);
                const successRates = currencies.map(c => c.success_rate);
                
                // Update chart
                updateCurrencyPerformanceChart(labels, successRates);
            } else {
                console.error('Error loading currency data:', data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching currency data:', error);
        });
}

// Update the currency performance chart with new data
function updateCurrencyPerformanceChart(labels, successRates) {
    const ctx = document.getElementById('currencyPerformanceChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (currencyPerformanceChart) {
        currencyPerformanceChart.destroy();
    }
    
    // Create new chart
    currencyPerformanceChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Success Rate',
                data: successRates,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgb(54, 162, 235)',
                pointBackgroundColor: 'rgb(54, 162, 235)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(54, 162, 235)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    min: 85,
                    max: 95
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.raw + '% success rate';
                        }
                    }
                }
            }
        }
    });
}

// Load market regime data
function loadMarketRegimeData() {
    // Fetch market regime data from API
    fetch('/api/analytics/market_regime_performance')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Extract data for chart
                const regimes = data.regime_performance;
                const labels = regimes.map(r => r.regime.charAt(0).toUpperCase() + r.regime.slice(1));
                const successRates = regimes.map(r => r.success_rate);
                
                // Update chart
                updateMarketRegimeChart(labels, successRates);
            } else {
                console.error('Error loading market regime data:', data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching market regime data:', error);
        });
}

// Update the market regime chart with new data
function updateMarketRegimeChart(labels, successRates) {
    const ctx = document.getElementById('marketRegimeChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (marketRegimeChart) {
        marketRegimeChart.destroy();
    }
    
    // Create new chart
    marketRegimeChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                label: 'Success Rate',
                data: successRates,
                backgroundColor: [
                    'rgba(40, 167, 69, 0.7)',
                    'rgba(23, 162, 184, 0.7)',
                    'rgba(220, 53, 69, 0.7)',
                    'rgba(255, 193, 7, 0.7)'
                ],
                borderColor: [
                    'rgba(40, 167, 69, 1)',
                    'rgba(23, 162, 184, 1)',
                    'rgba(220, 53, 69, 1)',
                    'rgba(255, 193, 7, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.raw + '% success rate';
                        }
                    }
                }
            }
        }
    });
}

// Load AI model performance data
function loadAiModelData(modelType) {
    // Fetch AI model data from API
    fetch(`/api/analytics/ai_model_performance?model_type=${modelType}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Extract data for chart
                const models = data.models.slice(0, 8); // Limit to top 8 for the chart
                const labels = models.map(m => m.name);
                const accuracies = models.map(m => m.accuracy);
                
                // Update chart
                updateAiModelsChart(labels, accuracies);
                
                // Update top models list
                updateTopAiModelsList(data.models.slice(0, 5)); // Top 5 for the list
            } else {
                console.error('Error loading AI model data:', data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching AI model data:', error);
        });
}

// Update the AI models chart with new data
function updateAiModelsChart(labels, accuracies) {
    const ctx = document.getElementById('aiModelsChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (aiModelsChart) {
        aiModelsChart.destroy();
    }
    
    // Create new chart
    aiModelsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Accuracy (%)',
                data: accuracies,
                backgroundColor: 'rgba(111, 66, 193, 0.7)',
                borderColor: 'rgba(111, 66, 193, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 80,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.raw + '%';
                        }
                    }
                }
            }
        }
    });
}

// Update the top AI models list with new data
function updateTopAiModelsList(models) {
    const listElement = document.getElementById('topAiModels');
    
    // Clear current list
    listElement.innerHTML = '';
    
    // Add each model to the list
    models.forEach(model => {
        // Determine badge color based on accuracy
        let badgeClass = 'bg-info';
        if (model.accuracy >= 90) {
            badgeClass = 'bg-success';
        } else if (model.accuracy < 85) {
            badgeClass = 'bg-warning';
        }
        
        // Create list item
        const listItem = document.createElement('li');
        listItem.className = 'list-group-item bg-dark text-light d-flex justify-content-between align-items-center';
        listItem.innerHTML = `
            <div>
                <strong>${model.name}</strong>
                <div class="small text-muted">${model.type}</div>
            </div>
            <span class="badge ${badgeClass}">${model.accuracy}%</span>
        `;
        
        // Add to list
        listElement.appendChild(listItem);
    });
}

// Load recent activity data
function loadRecentActivity() {
    // Fetch recent signals from API
    fetch('/api/analytics/recent_signals')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update recent activity table
                updateRecentActivityTable(data.signals);
            } else {
                console.error('Error loading recent activity:', data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching recent activity:', error);
        });
}

// Update the recent activity table with new data
function updateRecentActivityTable(signals) {
    const tableBody = document.getElementById('recentActivityTable');
    
    // Clear current table
    tableBody.innerHTML = '';
    
    // Add each signal to the table
    signals.forEach(signal => {
        // Determine signal badge class
        const signalBadgeClass = signal.signal_type === 'BUY' || signal.signal_type === 'CALL' ? 'bg-success' : 'bg-danger';
        
        // Determine status badge class
        let statusBadgeClass = 'bg-secondary';
        if (signal.status === 'VERIFIED') {
            statusBadgeClass = 'bg-success';
        } else if (signal.status === 'FAILED') {
            statusBadgeClass = 'bg-danger';
        } else if (signal.status === 'VERIFYING') {
            statusBadgeClass = 'bg-warning';
        }
        
        // Determine progress bar class
        let progressClass = 'bg-success';
        if (signal.probability < 85) {
            progressClass = 'bg-warning';
        }
        
        // Create table row
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${signal.currency}</td>
            <td><span class="badge ${signalBadgeClass}">${signal.signal_type}</span></td>
            <td>${signal.strategy}</td>
            <td>${signal.time_ago}</td>
            <td>
                <div class="progress" style="height: 10px;">
                    <div class="progress-bar ${progressClass}" style="width: ${signal.probability}%;">${signal.probability}%</div>
                </div>
            </td>
            <td><span class="badge ${statusBadgeClass}">${signal.status}</span></td>
        `;
        
        // Add to table
        tableBody.appendChild(row);
    });
}

// Load verification status data
function loadVerificationStatus() {
    // Fetch verification status from API
    fetch('/api/analytics/verification_status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update verification status list
                updateVerificationStatusList(data.active_verifications, data.verification_history);
            } else {
                console.error('Error loading verification status:', data.message);
            }
        })
        .catch(error => {
            console.error('Error fetching verification status:', error);
        });
}

// Update the verification status list with new data
function updateVerificationStatusList(activeVerifications, verificationHistory) {
    const statusList = document.getElementById('verificationStatusList');
    const noActiveMsg = document.getElementById('noActiveVerificationsMsg');
    
    // Clear current list
    statusList.innerHTML = '';
    
    // Check if there are active verifications
    if (activeVerifications.length === 0 && verificationHistory.length === 0) {
        // Show no active verifications message
        noActiveMsg.style.display = 'block';
        return;
    } else {
        // Hide no active verifications message
        noActiveMsg.style.display = 'none';
    }
    
    // Add active verifications
    activeVerifications.forEach(verification => {
        // Determine signal badge class
        const signalBadgeClass = verification.signal_type === 'BUY' || verification.signal_type === 'CALL' ? 'bg-success' : 'bg-danger';
        
        // Create verification card
        const card = document.createElement('div');
        card.className = 'p-3 border-bottom border-secondary';
        card.innerHTML = `
            <div class="d-flex justify-content-between align-items-center mb-2">
                <div>
                    <span class="badge ${signalBadgeClass} me-2">${verification.signal_type}</span>
                    <strong>${verification.currency_pair}</strong> 
                </div>
                <span class="badge bg-warning">IN PROGRESS</span>
            </div>
            <div class="mb-2">
                <div class="d-flex justify-content-between small mb-1">
                    <span>Verification Progress</span>
                    <span>${verification.checkpoints_passed}/${verification.checkpoints_total} checkpoints</span>
                </div>
                <div class="progress" style="height: 10px;">
                    <div class="progress-bar bg-primary" style="width: ${(verification.checkpoints_passed / verification.checkpoints_total) * 100}%"></div>
                </div>
            </div>
            <div>
                <div class="d-flex justify-content-between small mb-1">
                    <span>Current Score: ${verification.verification_score}%</span>
                    <span>Threshold: ${verification.threshold}%</span>
                </div>
                <div class="progress" style="height: 10px;">
                    <div class="progress-bar ${verification.verification_score >= verification.threshold ? 'bg-success' : 'bg-warning'}" style="width: ${verification.verification_score}%"></div>
                    <div class="progress-bar bg-transparent" style="width: 0%; border-right: 2px dashed yellow; height: 100%;"></div>
                </div>
            </div>
            <div class="d-flex justify-content-between small mt-2">
                <span>Started ${verification.elapsed_minutes} minutes ago</span>
                <span>Elapsed time: ${verification.elapsed_minutes} min</span>
            </div>
        `;
        
        // Add to list
        statusList.appendChild(card);
    });
    
    // Add verification history
    verificationHistory.forEach(verification => {
        // Determine signal badge class
        const signalBadgeClass = verification.signal_type === 'BUY' || verification.signal_type === 'CALL' ? 'bg-success' : 'bg-danger';
        
        // Determine status badge class
        let statusBadgeClass = 'bg-secondary';
        if (verification.status === 'VERIFIED' || verification.status === 'SUCCESS') {
            statusBadgeClass = 'bg-success';
        } else if (verification.status === 'FAILED' || verification.status === 'REJECTED') {
            statusBadgeClass = 'bg-danger';
        }
        
        // Create verification card
        const card = document.createElement('div');
        card.className = 'p-3 border-bottom border-secondary';
        card.innerHTML = `
            <div class="d-flex justify-content-between align-items-center mb-2">
                <div>
                    <span class="badge ${signalBadgeClass} me-2">${verification.signal_type}</span>
                    <strong>${verification.currency_pair}</strong> 
                </div>
                <span class="badge ${statusBadgeClass}">${verification.status}</span>
            </div>
            <div class="mb-2">
                <div class="d-flex justify-content-between small mb-1">
                    <span>Verification Progress</span>
                    <span>5/5 checkpoints</span>
                </div>
                <div class="progress" style="height: 10px;">
                    <div class="progress-bar bg-primary" style="width: 100%"></div>
                </div>
            </div>
            <div>
                <div class="d-flex justify-content-between small mb-1">
                    <span>Final Score: ${verification.final_score}%</span>
                    <span>Threshold: ${verification.threshold}%</span>
                </div>
                <div class="progress" style="height: 10px;">
                    <div class="progress-bar ${verification.final_score >= verification.threshold ? 'bg-success' : 'bg-danger'}" style="width: ${verification.final_score}%"></div>
                    <div class="progress-bar bg-transparent" style="width: 0%; border-right: 2px dashed yellow; height: 100%;"></div>
                </div>
            </div>
            <div class="d-flex justify-content-between small mt-2">
                <span>Completed ${verification.completed_ago}</span>
                <span>Elapsed time: ${verification.elapsed_minutes} min</span>
            </div>
        `;
        
        // Add to list
        statusList.appendChild(card);
    });
}