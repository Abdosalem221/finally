/**
 * Signals Chart JavaScript - Handles chart rendering and signal visualization
 */

// Chart colors
const chartColors = {
    green: 'rgba(75, 192, 192, 1)',
    greenLight: 'rgba(75, 192, 192, 0.2)',
    red: 'rgba(255, 99, 132, 1)',
    redLight: 'rgba(255, 99, 132, 0.2)',
    blue: 'rgba(54, 162, 235, 1)',
    blueLight: 'rgba(54, 162, 235, 0.2)',
    yellow: 'rgba(255, 205, 86, 1)',
    yellowLight: 'rgba(255, 205, 86, 0.2)',
    purple: 'rgba(153, 102, 255, 1)',
    purpleLight: 'rgba(153, 102, 255, 0.2)',
    orange: 'rgba(255, 159, 64, 1)',
    orangeLight: 'rgba(255, 159, 64, 0.2)'
};

// Initialize signal chart
function initSignalChart(containerId, currencyPair) {
    const container = document.getElementById(containerId);
    if (!container) return null;
    
    // Create canvas element
    const canvas = document.createElement('canvas');
    container.appendChild(canvas);
    
    // Initialize chart
    const ctx = canvas.getContext('2d');
    const chart = new Chart(ctx, {
        type: 'candlestick',
        data: {
            datasets: [{
                label: currencyPair,
                data: [] // Will be filled with price data
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'MMM d'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price'
                    }
                }
            }
        }
    });
    
    return chart;
}

// Load price data for a currency pair
function loadPriceData(chart, currencyPair, timeframe = '1d', limit = 30) {
    fetch(`/api/price_data?symbol=${currencyPair}&timeframe=${timeframe}&limit=${limit}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updatePriceChart(chart, data.price_data, currencyPair);
            } else {
                console.error('Error loading price data:', data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Update price chart with new data
function updatePriceChart(chart, priceData, currencyPair) {
    // Clear existing data
    chart.data.labels = [];
    chart.data.datasets = [];
    
    // Prepare data for chart
    const ohlcData = priceData.map(item => ({
        x: new Date(item.timestamp),
        o: item.open,
        h: item.high,
        l: item.low,
        c: item.close
    }));
    
    // Add candlestick dataset
    chart.data.datasets.push({
        label: currencyPair,
        data: ohlcData,
        color: {
            up: chartColors.green,
            down: chartColors.red,
            unchanged: chartColors.blue
        }
    });
    
    // Add volume dataset
    const volumeData = priceData.map(item => ({
        x: new Date(item.timestamp),
        y: item.volume || 0
    }));
    
    chart.data.datasets.push({
        label: 'Volume',
        data: volumeData,
        type: 'bar',
        backgroundColor: chartColors.blueLight,
        yAxisID: 'volume',
        order: 1
    });
    
    // Update chart options
    chart.options.scales.volume = {
        position: 'right',
        grid: {
            drawOnChartArea: false
        },
        beginAtZero: true
    };
    
    // Update chart
    chart.update();
}

// Add signal markers to chart
function addSignalsToChart(chart, signals) {
    if (!chart || !signals || signals.length === 0) return;
    
    // Create annotation plugin if needed
    if (!chart.options.plugins) {
        chart.options.plugins = {};
    }
    if (!chart.options.plugins.annotation) {
        chart.options.plugins.annotation = {
            annotations: {}
        };
    }
    
    // Clear existing annotations
    chart.options.plugins.annotation.annotations = {};
    
    // Add signals as annotations
    signals.forEach((signal, index) => {
        const signalDate = new Date(signal.timestamp);
        const isBuy = signal.signal_type === 'BUY' || signal.signal_type === 'CALL';
        const color = isBuy ? chartColors.green : chartColors.red;
        
        // Add signal line
        chart.options.plugins.annotation.annotations[`signal_${index}`] = {
            type: 'line',
            scaleID: 'x',
            value: signalDate,
            borderColor: color,
            borderWidth: 2,
            label: {
                content: signal.signal_type,
                enabled: true,
                position: 'top',
                backgroundColor: color
            }
        };
        
        // Add entry price line
        chart.options.plugins.annotation.annotations[`entry_${index}`] = {
            type: 'line',
            scaleID: 'y',
            value: signal.entry_price,
            borderColor: color,
            borderWidth: 1,
            borderDash: [5, 5],
            label: {
                content: 'Entry: ' + signal.entry_price.toFixed(4),
                enabled: true,
                position: 'left',
                backgroundColor: color
            }
        };
        
        // Add take profit line
        chart.options.plugins.annotation.annotations[`tp_${index}`] = {
            type: 'line',
            scaleID: 'y',
            value: signal.take_profit,
            borderColor: chartColors.green,
            borderWidth: 1,
            borderDash: [2, 2],
            label: {
                content: 'TP: ' + signal.take_profit.toFixed(4),
                enabled: true,
                position: 'right',
                backgroundColor: chartColors.green
            }
        };
        
        // Add stop loss line
        chart.options.plugins.annotation.annotations[`sl_${index}`] = {
            type: 'line',
            scaleID: 'y',
            value: signal.stop_loss,
            borderColor: chartColors.red,
            borderWidth: 1,
            borderDash: [2, 2],
            label: {
                content: 'SL: ' + signal.stop_loss.toFixed(4),
                enabled: true,
                position: 'right',
                backgroundColor: chartColors.red
            }
        };
    });
    
    // Update chart
    chart.update();
}

// Load signals for a currency pair
function loadSignals(chart, currencyPair) {
    fetch(`/api/currency_signals?symbol=${currencyPair}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                addSignalsToChart(chart, data.signals);
            } else {
                console.error('Error loading signals:', data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Load and display technical indicators
function loadTechnicalIndicators(containerId, currencyPair) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    fetch(`/api/technical_indicators?symbol=${currencyPair}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayTechnicalIndicators(container, data.indicators);
            } else {
                console.error('Error loading technical indicators:', data.message);
                container.innerHTML = `<div class="alert alert-danger">Error loading indicators: ${data.message}</div>`;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            container.innerHTML = '<div class="alert alert-danger">Error connecting to server</div>';
        });
}

// Display technical indicators in container
function displayTechnicalIndicators(container, indicators) {
    // Clear container
    container.innerHTML = '';
    
    if (!indicators || indicators.length === 0) {
        container.innerHTML = '<div class="alert alert-info">No indicators available</div>';
        return;
    }
    
    // Create table for indicators
    const table = document.createElement('table');
    table.className = 'table table-striped table-hover';
    
    // Create table header
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>Indicator</th>
            <th>Value</th>
            <th>Signal</th>
        </tr>
    `;
    table.appendChild(thead);
    
    // Create table body
    const tbody = document.createElement('tbody');
    indicators.forEach(indicator => {
        const row = document.createElement('tr');
        
        // Determine signal style
        let signalClass = '';
        if (indicator.signal === 'BUY') {
            signalClass = 'text-success';
        } else if (indicator.signal === 'SELL') {
            signalClass = 'text-danger';
        } else {
            signalClass = 'text-secondary';
        }
        
        row.innerHTML = `
            <td>${indicator.name}</td>
            <td>${indicator.value}</td>
            <td class="${signalClass}">${indicator.signal}</td>
        `;
        
        tbody.appendChild(row);
    });
    table.appendChild(tbody);
    
    // Add table to container
    container.appendChild(table);
}

// Generate a new signal
function generateSignal(currencyPair, marketType, callback) {
    fetch(`/api/generate_signal?currency=${currencyPair}&market_type=${marketType}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (callback && typeof callback === 'function') {
                    callback(data.signal);
                }
            } else {
                console.error('Error generating signal:', data.message);
                alert('Error generating signal: ' + (data.message || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error connecting to server');
        });
}

// Calculate risk/reward ratio
function calculateRiskReward(signal) {
    if (!signal || !signal.entry_price || !signal.take_profit || !signal.stop_loss) {
        return null;
    }
    
    let risk, reward;
    
    if (signal.signal_type === 'BUY' || signal.signal_type === 'CALL') {
        risk = signal.entry_price - signal.stop_loss;
        reward = signal.take_profit - signal.entry_price;
    } else {
        risk = signal.stop_loss - signal.entry_price;
        reward = signal.entry_price - signal.take_profit;
    }
    
    if (risk <= 0 || reward <= 0) {
        return null;
    }
    
    return reward / risk;
}

// Format risk/reward ratio for display
function formatRiskReward(ratio) {
    if (ratio === null || isNaN(ratio)) {
        return 'N/A';
    }
    
    return `1:${ratio.toFixed(2)}`;
}
