/**
 * Alerts JavaScript - Handles alert creation and management
 */

// Create a new price alert
function createAlert(alertData, callback) {
    fetch('/api/create_alert', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(alertData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            if (callback && typeof callback === 'function') {
                callback(data.alert);
            }
        } else {
            console.error('Error creating alert:', data.message);
            alert('Error creating alert: ' + (data.message || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error connecting to server');
    });
}

// Delete an existing alert
function deleteAlert(alertId, callback) {
    fetch(`/api/delete_alert/${alertId}`, {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            if (callback && typeof callback === 'function') {
                callback(alertId);
            }
        } else {
            console.error('Error deleting alert:', data.message);
            alert('Error deleting alert: ' + (data.message || 'Unknown error'));
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error connecting to server');
    });
}

// Load active alerts
function loadActiveAlerts(callback) {
    fetch('/api/active_alerts')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (callback && typeof callback === 'function') {
                    callback(data.alerts);
                }
            } else {
                console.error('Error loading alerts:', data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Format alert object for display
function formatAlertMessage(alert) {
    let message = '';
    
    if (alert.alert_type === 'price') {
        message = `Price level alert at ${alert.price_level} for ${alert.currency_symbol}`;
    } else if (alert.alert_type === 'pattern') {
        message = `Pattern recognition alert for ${alert.currency_symbol}: ${alert.pattern || 'Custom pattern'}`;
    } else if (alert.alert_type === 'support_resistance') {
        message = `Support/Resistance level alert at ${alert.price_level} for ${alert.currency_symbol}`;
    } else {
        message = alert.message || `Alert for ${alert.currency_symbol}`;
    }
    
    return message;
}

// Create price level alert form elements
function initPriceLevelAlertForm() {
    const alertForm = document.getElementById('alert-form');
    if (!alertForm) return;
    
    // Get current price to help user set alerts
    const currencySelect = document.getElementById('alert-currency');
    currencySelect.addEventListener('change', function() {
        if (!this.value) return;
        
        // Get current price for selected currency
        fetch(`/api/current_price?symbol=${this.value}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const priceInput = document.getElementById('alert-price');
                    const price = data.price;
                    
                    // Set price as placeholder
                    priceInput.placeholder = `Current price: ${price}`;
                    
                    // Suggest alert message
                    const alertType = document.getElementById('alert-type').value;
                    const messageInput = document.getElementById('alert-message');
                    
                    if (alertType === 'price' && messageInput.value === '') {
                        messageInput.value = `Price level alert for ${this.value}`;
                    }
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    });
    
    // Change alert message based on alert type
    const alertType = document.getElementById('alert-type');
    alertType.addEventListener('change', function() {
        const currency = document.getElementById('alert-currency').value;
        const messageInput = document.getElementById('alert-message');
        
        if (!currency) return;
        
        if (this.value === 'price') {
            messageInput.value = `Price level alert for ${currency}`;
        } else if (this.value === 'pattern') {
            messageInput.value = `Pattern recognition alert for ${currency}`;
        } else if (this.value === 'support_resistance') {
            messageInput.value = `Support/Resistance level alert for ${currency}`;
        }
    });
}

// Show browser notifications for triggered alerts
function showAlertNotification(alert) {
    // Check if browser notifications are supported
    if (!("Notification" in window)) {
        console.log("This browser does not support desktop notification");
        return;
    }
    
    // Check if permission is granted
    if (Notification.permission === "granted") {
        createNotification(alert);
    }
    // Otherwise, ask for permission
    else if (Notification.permission !== "denied") {
        Notification.requestPermission().then(function (permission) {
            if (permission === "granted") {
                createNotification(alert);
            }
        });
    }
}

// Create a notification for an alert
function createNotification(alert) {
    const title = `Alert: ${alert.currency_symbol}`;
    const options = {
        body: formatAlertMessage(alert),
        icon: '/static/images/alert-icon.png',
        badge: '/static/images/alert-badge.png'
    };
    
    const notification = new Notification(title, options);
    
    // Close notification after 5 seconds
    setTimeout(notification.close.bind(notification), 5000);
    
    // Handle notification click
    notification.onclick = function() {
        window.focus();
        notification.close();
    };
}

// Initialize alert UI components
document.addEventListener('DOMContentLoaded', function() {
    initPriceLevelAlertForm();
    
    // Request notification permission if needed
    if ("Notification" in window && Notification.permission !== "denied") {
        Notification.requestPermission();
    }
});
