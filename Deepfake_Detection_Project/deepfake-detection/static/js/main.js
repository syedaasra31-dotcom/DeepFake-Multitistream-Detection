/* ============================================
   DeepGuard - Deepfake Detection System
   Main JavaScript
   ============================================ */

(function () {
    'use strict';

    /* --- Scroll-Triggered Fade-In Animations --- */
    function initScrollAnimations() {
        var animatedElements = document.querySelectorAll(
            '.fade-in, .fade-in-left, .fade-in-right, .scale-in'
        );
        if (!animatedElements.length) return;

        var observerOptions = {
            root: null,
            rootMargin: '0px 0px -80px 0px',
            threshold: 0.1
        };

        var observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target);
                }
            });
        }, observerOptions);

        animatedElements.forEach(function (el) {
            observer.observe(el);
        });
    }

    /* --- Navbar Scroll Effect --- */
    function initNavbarScroll() {
        var navbar = document.getElementById('mainNav');
        if (!navbar) return;

        var lastScrollY = 0;
        var ticking = false;

        function onScroll() {
            lastScrollY = window.scrollY;
            if (!ticking) {
                window.requestAnimationFrame(function () {
                    if (lastScrollY > 50) {
                        navbar.classList.add('scrolled');
                    } else {
                        navbar.classList.remove('scrolled');
                    }
                    ticking = false;
                });
                ticking = true;
            }
        }

        window.addEventListener('scroll', onScroll, { passive: true });
        onScroll();
    }

    /* --- Stat Counter Animation --- */
    function initStatCounters() {
        var counters = document.querySelectorAll('[data-count]');
        if (!counters.length) return;

        var observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    animateCounter(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.5 });

        counters.forEach(function (counter) {
            observer.observe(counter);
        });
    }

    function animateCounter(element) {
        var target = parseInt(element.getAttribute('data-count'), 10);
        var duration = 2000;
        var start = 0;
        var startTime = null;
        var suffix = element.getAttribute('data-suffix') || '';
        var prefix = element.getAttribute('data-prefix') || '';
        var decimals = parseInt(element.getAttribute('data-decimals') || '0', 10);

        function easeOutQuart(t) {
            return 1 - Math.pow(1 - t, 4);
        }

        function step(timestamp) {
            if (!startTime) startTime = timestamp;
            var progress = Math.min((timestamp - startTime) / duration, 1);
            var easedProgress = easeOutQuart(progress);
            var current = start + (target - start) * easedProgress;

            element.textContent = prefix + current.toFixed(decimals) + suffix;

            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        }

        window.requestAnimationFrame(step);
    }

    /* --- File Upload Drag & Drop --- */
    function initFileUpload() {
        var dropzone = document.getElementById('uploadDropzone');
        var fileInput = document.getElementById('fileInput');
        if (!dropzone || !fileInput) return;

        var allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/x-ms-wmv', 'video/webm'];
        var allowedExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.webm'];
        var maxFileSize = 500 * 1024 * 1024; // 500MB

        dropzone.addEventListener('click', function () {
            fileInput.click();
        });

        dropzone.addEventListener('dragenter', function (e) {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-over');
        });

        dropzone.addEventListener('dragover', function (e) {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('drag-over');
        });

        dropzone.addEventListener('dragleave', function (e) {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-over');
        });

        dropzone.addEventListener('drop', function (e) {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('drag-over');

            var files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });

        fileInput.addEventListener('change', function () {
            if (fileInput.files.length > 0) {
                handleFileSelect(fileInput.files[0]);
            }
        });

        function handleFileSelect(file) {
            // Validate extension
            var fileName = file.name.toLowerCase();
            var ext = fileName.substring(fileName.lastIndexOf('.'));
            if (allowedExtensions.indexOf(ext) === -1) {
                dropzone.classList.add('file-error');
                dropzone.classList.remove('file-selected');
                if (typeof showToast === 'function') {
                    showToast('Invalid file type. Please upload: ' + allowedExtensions.join(', '), 'error');
                }
                return;
            }

            // Validate MIME type (if available)
            if (file.type && allowedTypes.indexOf(file.type) === -1) {
                dropzone.classList.add('file-error');
                dropzone.classList.remove('file-selected');
                if (typeof showToast === 'function') {
                    showToast('Unsupported video format. Please try a different file.', 'error');
                }
                return;
            }

            // Validate file size
            if (file.size > maxFileSize) {
                dropzone.classList.add('file-error');
                dropzone.classList.remove('file-selected');
                if (typeof showToast === 'function') {
                    showToast('File too large. Maximum size is 500MB.', 'error');
                }
                return;
            }

            dropzone.classList.remove('file-error');
            dropzone.classList.add('file-selected');

            var icon = dropzone.querySelector('.dropzone-icon');
            var title = dropzone.querySelector('.dropzone-title');
            var subtitle = dropzone.querySelector('.dropzone-subtitle');

            if (icon) icon.className = 'fas fa-check-circle dropzone-icon';
            if (title) title.textContent = file.name;
            if (subtitle) {
                var sizeMB = (file.size / (1024 * 1024)).toFixed(1);
                subtitle.textContent = sizeMB + ' MB \u2022 Ready to analyze';
            }

            if (typeof showToast === 'function') {
                showToast('File selected: ' + file.name, 'success');
            }
        }
    }

    /* --- Smooth Scroll for Anchor Links --- */
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(function (link) {
            link.addEventListener('click', function (e) {
                var href = this.getAttribute('href');
                if (href === '#') return;

                var target = document.querySelector(href);
                if (target) {
                    e.preventDefault();
                    var offsetTop = target.getBoundingClientRect().top + window.pageYOffset - 80;
                    window.scrollTo({
                        top: offsetTop,
                        behavior: 'smooth'
                    });
                }
            });
        });
    }

    /* --- Active Nav Link Highlighting --- */
    function initActiveNavLink() {
        var navLinks = document.querySelectorAll('.navbar-dark-custom .nav-link');
        if (!navLinks.length) return;

        var currentPath = window.location.pathname;
        navLinks.forEach(function (link) {
            var href = link.getAttribute('href');
            if (href && currentPath === href) {
                link.classList.add('active');
            }
        });

        // Highlight on scroll for same-page sections
        var sections = document.querySelectorAll('section[id]');
        if (!sections.length) return;

        var sectionObserver = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    var id = entry.target.getAttribute('id');
                    navLinks.forEach(function (link) {
                        link.classList.remove('active');
                        var linkHref = link.getAttribute('href');
                        if (linkHref && linkHref.indexOf('#' + id) !== -1) {
                            link.classList.add('active');
                        }
                    });
                }
            });
        }, { rootMargin: '-30% 0px -60% 0px' });

        sections.forEach(function (section) {
            sectionObserver.observe(section);
        });
    }

    /* --- Chart.js Default Theme Configuration --- */
    function initChartTheme() {
        if (typeof Chart === 'undefined') return;

        Chart.defaults.color = '#8892b0';
        Chart.defaults.borderColor = 'rgba(35, 53, 84, 0.5)';
        Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";
        Chart.defaults.font.size = 12;
        Chart.defaults.plugins.legend.labels.color = '#ccd6f6';
        Chart.defaults.plugins.legend.labels.padding = 16;
        Chart.defaults.plugins.legend.labels.usePointStyle = true;
        Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;
        Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(17, 34, 64, 0.95)';
        Chart.defaults.plugins.tooltip.titleColor = '#e6f1ff';
        Chart.defaults.plugins.tooltip.bodyColor = '#ccd6f6';
        Chart.defaults.plugins.tooltip.borderColor = 'rgba(100, 255, 218, 0.2)';
        Chart.defaults.plugins.tooltip.borderWidth = 1;
        Chart.defaults.plugins.tooltip.cornerRadius = 8;
        Chart.defaults.plugins.tooltip.padding = 12;
        Chart.defaults.plugins.tooltip.displayColors = true;
        Chart.defaults.plugins.tooltip.boxPadding = 4;
        Chart.defaults.scale.grid = {
            color: 'rgba(35, 53, 84, 0.3)',
            drawBorder: false
        };
        Chart.defaults.scale.ticks = {
            color: '#8892b0',
            padding: 8
        };
    }

    /* --- Tooltip Initialization (CSS custom tooltips) --- */
    function initTooltips() {
        var tooltipEls = document.querySelectorAll('[data-tooltip]');
        tooltipEls.forEach(function (el) {
            if (!el.classList.contains('tooltip-custom')) {
                el.classList.add('tooltip-custom');
            }
        });
    }

    /* --- Circular Gauge Animation --- */
    function initCircularGauges() {
        var gauges = document.querySelectorAll('.gauge-fill[data-percent]');
        if (!gauges.length) return;

        var observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    var gauge = entry.target;
                    var percent = parseFloat(gauge.getAttribute('data-percent'));
                    var circumference = 2 * Math.PI * 60; // r=60
                    var offset = circumference - (percent / 100) * circumference;
                    gauge.style.strokeDashoffset = offset;
                    observer.unobserve(gauge);
                }
            });
        }, { threshold: 0.5 });

        gauges.forEach(function (gauge) {
            observer.observe(gauge);
        });
    }

    /* --- Loading Overlay --- */
    window.showLoading = function () {
        var overlay = document.getElementById('loadingOverlay');
        if (overlay) overlay.classList.add('active');
    };
    window.hideLoading = function () {
        var overlay = document.getElementById('loadingOverlay');
        if (overlay) overlay.classList.remove('active');
    };

    /* --- Set Current Year --- */
    function setCurrentYear() {
        var yearEl = document.getElementById('currentYear');
        if (yearEl) {
            yearEl.textContent = new Date().getFullYear();
        }
    }

    /* --- Close Mobile Navbar on Link Click --- */
    function initMobileNavClose() {
        document.querySelectorAll('.navbar-nav .nav-link').forEach(function (link) {
            link.addEventListener('click', function () {
                var navCollapse = document.querySelector('.navbar-collapse');
                if (navCollapse && navCollapse.classList.contains('show')) {
                    var bsCollapse = bootstrap.Collapse.getInstance(navCollapse);
                    if (bsCollapse) bsCollapse.hide();
                }
            });
        });
    }

    /* --- Initialize Everything --- */
    document.addEventListener('DOMContentLoaded', function () {
        setCurrentYear();
        initNavbarScroll();
        initScrollAnimations();
        initStatCounters();
        initFileUpload();
        initSmoothScroll();
        initActiveNavLink();
        initChartTheme();
        initTooltips();
        initCircularGauges();
        initMobileNavClose();
    });

})();
