
// Service Worker للتحميل السريع
const CACHE_NAME = 'algo-trader-pro-v1';
const urlsToCache = [
  '/',
  '/static/css/style.css',
  '/static/js/dashboard.js',
  '/static/js/signals_chart.js'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});
