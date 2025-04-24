// تهيئة التطبيق
document.addEventListener('DOMContentLoaded', () => {
    console.log('App initialized');
    initializeApp();
    setupEventListeners();
    initializeTheme();
    initializeRealTimeUpdates();
});

// تهيئة التطبيق
function initializeApp() {
    checkAuthStatus();
    
    // التحقق من وجود الدوال قبل استدعائها
    if (typeof initializeCharts === 'function') {
        initializeCharts();
    }
    
    // تحميل البيانات الأولية
    loadDashboardData();
    loadLatestSignals();
    loadWatchlist();
    loadRecentAlerts();
}

// تحميل بيانات لوحة التحكم
async function loadDashboardData() {
    try {
        const response = await fetch('/api/dashboard_data');
        const data = await response.json();
        if (data.success) {
            updateDashboardStats(data.data);
            await updateCharts(data.data);
        }
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showNotification('error', 'فشل تحميل البيانات');
    }
}

// تحميل الإشارات الأخيرة
async function loadLatestSignals() {
    try {
        const response = await fetch('/api/latest_signals');
        const data = await response.json();
        if (data.success) {
            updateSignalsTable(data.signals);
        }
    } catch (error) {
        console.error('Error loading signals:', error);
    }
}

// تحميل قائمة المراقبة
async function loadWatchlist() {
    try {
        const response = await fetch('/api/watchlist');
        const data = await response.json();
        if (data.success) {
            updateWatchlist(data.currencies);
        }
    } catch (error) {
        console.error('Error loading watchlist:', error);
    }
}

// تحميل التنبيهات الأخيرة
async function loadRecentAlerts() {
    try {
        const response = await fetch('/api/recent_alerts');
        const data = await response.json();
        if (data.success) {
            updateAlertsPanel(data.alerts);
        }
    } catch (error) {
        console.error('Error loading alerts:', error);
    }
}

// تحديث إحصائيات لوحة التحكم
function updateDashboardStats(data) {
    const statsElements = {
        'active-signals': data.active_signals,
        'success-rate': data.success_rate + '%',
        'monitored-pairs': data.monitored_pairs,
        'active-alerts': data.active_alerts
    };

    for (const [id, value] of Object.entries(statsElements)) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }
}

// تحديث جدول الإشارات
function updateSignalsTable(signals) {
    const tableBody = document.getElementById('signals-table-body');
    if (!tableBody) return;

    tableBody.innerHTML = '';
    signals.forEach(signal => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${signal.currency_pair}</td>
            <td class="${signal.signal_type === 'BUY' ? 'text-success' : 'text-danger'}">
                ${signal.signal_type}
            </td>
            <td>${signal.entry_price}</td>
            <td>${signal.take_profit}</td>
            <td>${signal.stop_loss}</td>
            <td>${new Date(signal.timestamp).toLocaleString()}</td>
        `;
        tableBody.appendChild(row);
    });
}

// تحديث قائمة المراقبة
function updateWatchlist(currencies) {
    const watchlist = document.getElementById('watchlist');
    if (!watchlist) return;

    watchlist.innerHTML = '';
    currencies.forEach(currency => {
        const item = document.createElement('div');
        item.className = 'watchlist-item';
        item.innerHTML = `
            <span>${currency.symbol}</span>
            <span class="${currency.change >= 0 ? 'text-success' : 'text-danger'}">
                ${currency.price} (${currency.change}%)
            </span>
        `;
        watchlist.appendChild(item);
    });
}

// تحديث لوحة التنبيهات
function updateAlertsPanel(alerts) {
    const alertsList = document.getElementById('alerts-list');
    if (!alertsList) return;

    alertsList.innerHTML = '';
    alerts.forEach(alert => {
        const item = document.createElement('div');
        item.className = 'alert-item';
        item.innerHTML = `
            <div class="alert-header">
                <span>${alert.currency_symbol}</span>
                <span>${new Date(alert.created_at).toLocaleString()}</span>
            </div>
            <div class="alert-body">${alert.message}</div>
        `;
        alertsList.appendChild(item);
    });
}

// إعداد مستمعي الأحداث
function setupEventListeners() {
    // مستمع تبديل السمة
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }

    // مستمعي النماذج
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', handleFormSubmit);
    });

    // مستمعي التنبيهات
    setupAlertListeners();
}

// التحقق من حالة المصادقة
async function checkAuthStatus() {
    fetch('/api/auth/status')
        .then(response => {
            if (!response.ok) {
                console.log('Auth check failed');
                updateUIBasedOnAuth(false);
                return;
            }
            return response.json();
        })
        .then(data => {
            updateUIBasedOnAuth(data.isAuthenticated);
        })
        .catch(error => {
            console.error('Auth check failed:', error);
            handleError(error);
        });
}

// تحديث واجهة المستخدم بناءً على حالة المصادقة
function updateUIBasedOnAuth(isAuthenticated) {
    const authElements = document.querySelectorAll('.auth-dependent');
    authElements.forEach(element => {
        element.style.display = isAuthenticated ? 'block' : 'none';
    });
}

// تهيئة السمة
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

// تبديل السمة
function toggleTheme() {
    const currentTheme = document.body.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

// تحديث أيقونة السمة
function updateThemeIcon(theme) {
    const icon = document.querySelector('#theme-toggle i');
    if (icon) {
        icon.className = theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
    }
}

// معالجة النماذج
async function handleFormSubmit(event) {
    event.preventDefault();
    const form = event.target;

    try {
        const formData = new FormData(form);
        const response = await fetch(form.action, {
            method: form.method,
            body: formData
        });

        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();

        if (data.success) {
            showNotification('success', data.message);
        } else {
            showNotification('error', data.message);
        }
    } catch (error) {
        console.error('Form submission error:', error);
        handleError(error);
    }
}

// إظهار الإشعارات
function showNotification(type, message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.querySelector('main').prepend(alertDiv);
}

// معالجة الأخطاء
function handleError(error) {
    console.error('Error:', error);
    showNotification('error', 'حدث خطأ. يرجى المحاولة مرة أخرى.');
}

// تهيئة التحديثات في الوقت الفعلي
function initializeRealTimeUpdates() {
    // تكوين WebSocket
    const ws = new WebSocket(`wss://${window.location.host}/ws`);

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleRealtimeUpdate(data);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// معالجة التحديثات في الوقت الفعلي
function handleRealtimeUpdate(data) {
    switch (data.type) {
        case 'price_update':
            updatePriceDisplay(data);
            break;
        case 'signal':
            handleNewSignal(data);
            break;
        case 'alert':
            handleNewAlert(data);
            break;
    }
}

// تحديث عرض الأسعار
function updatePriceDisplay(data) {
    const priceElement = document.querySelector(`[data-currency="${data.currency}"]`);
    if (priceElement) {
        priceElement.textContent = data.price;
        priceElement.classList.add('price-update');
        setTimeout(() => priceElement.classList.remove('price-update'), 1000);
    }
}

// معالجة الإشارات الجديدة
function handleNewSignal(data) {
    // تحديث جدول الإشارات
    const signalsTable = document.getElementById('signals-table-body');
    if (signalsTable) {
        const row = createSignalRow(data);
        signalsTable.prepend(row);
        row.classList.add('new-signal');
        setTimeout(() => row.classList.remove('new-signal'), 1000);
    }
}

// إنشاء صف إشارة
function createSignalRow(signal) {
    const row = document.createElement('tr');
    row.innerHTML = `
        <td>${signal.currency_pair}</td>
        <td class="${signal.type === 'buy' ? 'text-success' : 'text-danger'}">${signal.type}</td>
        <td>${signal.entry_price}</td>
        <td>${signal.target_price}</td>
        <td>${signal.stop_loss}</td>
        <td>${signal.timestamp}</td>
    `;
    return row;
}

// معالجة التنبيهات الجديدة
function handleNewAlert(data) {
    // إضافة تنبيه جديد
    const alertsList = document.getElementById('alerts-list');
    if (alertsList) {
        const alertItem = createAlertItem(data);
        alertsList.prepend(alertItem);
        alertItem.classList.add('new-alert');
        setTimeout(() => alertItem.classList.remove('new-alert'), 1000);
    }
}

// إنشاء عنصر تنبيه
function createAlertItem(alert) {
    const li = document.createElement('li');
    li.className = 'list-group-item';
    li.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <span>${alert.message}</span>
            <small class="text-muted">${alert.timestamp}</small>
        </div>
    `;
    return li;
}

document.addEventListener('DOMContentLoaded', function() {
    if (typeof initializeCharts === 'function') {
        initializeCharts();
    }
});