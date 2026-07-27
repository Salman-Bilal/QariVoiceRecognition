/**
 * Qari Voice Recognition — Frontend Application Logic
 * Handles audio upload, recording, API communication, and results rendering
 */

const API_BASE = 'http://localhost:8000';

// ─── Utility Functions ────────────────────────────────────────────────────────

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast show ${type}`;
    setTimeout(() => { toast.className = 'toast'; }, 4000);
}

function setLoading(prefix, visible, stepText = '') {
    const loadingEl = document.getElementById(`${prefix}-loading`);
    const submitBtn = document.getElementById(`${prefix}-submit-btn`);
    if (loadingEl) {
        loadingEl.style.display = visible ? 'block' : 'none';
        if (stepText) {
            const stepEl = document.getElementById(`${prefix}-loading-step`);
            if (stepEl) stepEl.textContent = stepText;
        }
    }
    if (submitBtn) submitBtn.disabled = visible;
}

function formatScore(score) {
    return `${score}/100`;
}

function getScoreColor(score) {
    if (score >= 75) return '#10b981'; // green  — high similarity
    if (score >= 45) return '#3b82f6'; // blue   — medium similarity
    if (score >= 20) return '#f59e0b'; // yellow — low similarity
    return '#ef4444';                  // red    — no similarity
}

// ─── Tab Switching ─────────────────────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const target = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(target).classList.add('active');
    });
});

// ─── Smooth Scroll Nav Links ────────────────────────────────────────────────────

document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const id = link.getAttribute('href').slice(1);
        const section = document.getElementById(id);
        if (section) section.scrollIntoView({ behavior: 'smooth' });

        // Also activate tab if it's a tab section
        const tabBtn = document.querySelector(`.tab-btn[data-tab="${id}"]`);
        if (tabBtn) tabBtn.click();
    });
});

// ─── Generic File Upload Setup ──────────────────────────────────────────────────

function setupFileUpload(prefix) {
    const uploadArea = document.getElementById(`${prefix}-upload-area`);
    const fileInput = document.getElementById(`${prefix}-file-input`);
    const fileInfo = document.getElementById(`${prefix}-file-info`);
    const fileName = document.getElementById(`${prefix}-file-name`);
    const removeBtn = document.getElementById(`${prefix}-remove-file`);
    const submitBtn = document.getElementById(`${prefix}-submit-btn`);

    let selectedFile = null;

    function handleFile(file) {
        const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/x-m4a', 'audio/flac', 'audio/x-flac'];
        const allowedExts = ['.wav', '.mp3', '.m4a', '.flac'];
        const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

        if (!allowedExts.includes(ext)) {
            showToast('Invalid file type. Please upload WAV, MP3, M4A, or FLAC.', 'error');
            return;
        }
        if (file.size > 50 * 1024 * 1024) {
            showToast('File too large. Maximum size is 50MB.', 'error');
            return;
        }

        selectedFile = file;
        fileName.textContent = `${file.name}  (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        fileInfo.style.display = 'flex';
        submitBtn.disabled = false;
        showToast('File loaded successfully!', 'success');
    }

    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    if (removeBtn) {
        removeBtn.addEventListener('click', () => {
            selectedFile = null;
            fileInput.value = '';
            fileInfo.style.display = 'none';
            submitBtn.disabled = true;

            // Hide results
            const resultsEl = document.getElementById(`${prefix}-results`);
            if (resultsEl) resultsEl.style.display = 'none';
        });
    }

    return { getFile: () => selectedFile };
}

// ─── Recording Setup ────────────────────────────────────────────────────────────

function setupRecording(prefix) {
    const recordBtn = document.getElementById(`${prefix}-record-btn`);
    const timerEl = document.getElementById(`${prefix}-timer`);
    const previewEl = document.getElementById(`${prefix}-audio-preview`);
    const audioPlayer = document.getElementById(`${prefix}-audio-player`);
    const submitBtn = document.getElementById(`${prefix}-submit-btn`);

    if (!recordBtn) return { getRecordedFile: () => null };

    let mediaRecorder = null;
    let chunks = [];
    let timerInterval = null;
    let seconds = 0;
    let recordedBlob = null;

    function formatTime(s) {
        const m = Math.floor(s / 60).toString().padStart(2, '0');
        const sec = (s % 60).toString().padStart(2, '0');
        return `${m}:${sec}`;
    }

    recordBtn.addEventListener('click', async () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            // Stop recording
            mediaRecorder.stop();
            clearInterval(timerInterval);
            recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
            recordBtn.classList.remove('recording');
        } else {
            // Start recording
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                chunks = [];
                seconds = 0;
                timerEl.textContent = '00:00';

                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
                mediaRecorder.onstop = () => {
                    recordedBlob = new Blob(chunks, { type: 'audio/wav' });
                    const url = URL.createObjectURL(recordedBlob);
                    audioPlayer.src = url;
                    previewEl.style.display = 'block';
                    submitBtn.disabled = false;
                    stream.getTracks().forEach(t => t.stop());
                    showToast('Recording saved! Click Identify Qari to analyze.', 'success');
                };

                mediaRecorder.start();
                recordBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Recording';
                recordBtn.classList.add('recording');

                timerInterval = setInterval(() => {
                    seconds++;
                    timerEl.textContent = formatTime(seconds);
                    if (seconds >= 300) {
                        mediaRecorder.stop();
                        clearInterval(timerInterval);
                    }
                }, 1000);

            } catch (err) {
                showToast('Microphone access denied. Please allow mic access.', 'error');
            }
        }
    });

    return {
        getRecordedFile: () => {
            if (!recordedBlob) return null;
            return new File([recordedBlob], 'recorded_recitation.wav', { type: 'audio/wav' });
        }
    };
}

// ─── Load dropdowns from API ────────────────────────────────────────────────────

async function loadDropdowns() {
    try {
        const [qariRes, surahRes] = await Promise.all([
            fetch(`${API_BASE}/api/list-qaris`),
            fetch(`${API_BASE}/api/available-surahs`)
        ]);

        const qariData = await qariRes.json();
        const surahData = await surahRes.json();

        const qariSelect = document.getElementById('qari-select');
        const surahSelect = document.getElementById('surah-select');

        if (qariData.success && qariSelect) {
            qariSelect.innerHTML = '';
            qariData.qaris.forEach(q => {
                const opt = document.createElement('option');
                opt.value = q;
                opt.textContent = q;
                if (q === qariData.default_reference) opt.selected = true;
                qariSelect.appendChild(opt);
            });
        }

        if (surahData.success && surahSelect) {
            surahSelect.innerHTML = '';
            surahData.surahs.forEach(s => {
                const opt = document.createElement('option');
                opt.value = s;
                opt.textContent = s.replace(/_/g, ' ');
                surahSelect.appendChild(opt);
            });
        }
    } catch (err) {
        console.warn('Could not load dropdown data from API:', err);
    }
}

// ─── TOP-K SLIDER ─────────────────────────────────────────────────────────────

const topKSlider = document.getElementById('top-k-slider');
const topKValue = document.getElementById('top-k-value');
if (topKSlider) {
    topKSlider.addEventListener('input', () => {
        topKValue.textContent = topKSlider.value;
    });
}

// ─── Identify Qari ─────────────────────────────────────────────────────────────

const identifyUpload = setupFileUpload('identify');
const identifyRecorder = setupRecording('identify');

let identifyChart = null;

document.getElementById('identify-submit-btn').addEventListener('click', async () => {
    const file = identifyUpload.getFile() || identifyRecorder.getRecordedFile();
    if (!file) { showToast('Please select or record an audio file.', 'error'); return; }

    const topK = parseInt(document.getElementById('top-k-slider').value);

    const formData = new FormData();
    formData.append('audio_file', file);
    formData.append('top_k', topK);

    setLoading('identify', true, 'Extracting voice features…');
    document.getElementById('identify-results').style.display = 'none';

    try {
        const res = await fetch(`${API_BASE}/api/identify-qari`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Server error');
        }

        const data = await res.json();
        renderIdentifyResults(data, topK);
        showToast('Analysis complete!', 'success');

    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    } finally {
        setLoading('identify', false);
    }
});

function renderIdentifyResults(data, topK) {
    const resultsEl = document.getElementById('identify-results');
    resultsEl.style.display = 'block';
    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Top match hero
    document.getElementById('result-top-qari').textContent = `🏆 ${data.top_match.qari}`;
    document.getElementById('result-top-similarity').textContent =
        `${data.top_match.similarity}% similarity — ${data.top_match.confidence.toUpperCase()} confidence`;

    // Bar chart
    const allMatches = data.all_matches;
    const labels = allMatches.map(m => m.qari);
    const values = allMatches.map(m => m.similarity);
    const colors = values.map(v => getScoreColor(v) + 'cc');

    if (identifyChart) identifyChart.destroy();
    identifyChart = new Chart(document.getElementById('identify-chart'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Similarity (%)',
                data: values,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('cc', '')),
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Voice Similarity Against All Qaris',
                    font: { size: 16, weight: 'bold' }
                }
            },
            scales: {
                y: { min: 0, max: 100, title: { display: true, text: 'Similarity (%)' } },
                x: { ticks: { maxRotation: 45 } }
            }
        }
    });

    // Ranked list — show top_k results
    const matchesList = document.getElementById('matches-list');
    matchesList.innerHTML = '';
    data.top_k_matches.forEach((match, i) => {
        const item = document.createElement('div');
        item.className = 'match-item';
        // Confidence label: based on similarity_percent
        let confidenceLabel = 'Low';
        let confidenceColor = '#ef4444';
        if (match.similarity >= 75) { confidenceLabel = 'High';   confidenceColor = '#10b981'; }
        else if (match.similarity >= 45) { confidenceLabel = 'Medium'; confidenceColor = '#f59e0b'; }

        item.innerHTML = `
            <div class="match-rank">#${i + 1}</div>
            <div class="match-name">${match.qari}</div>
            <div class="match-bar-wrap" style="flex:1;background:#e2e8f0;height:8px;border-radius:4px;margin:0 12px">
                <div style="width:${match.similarity}%;height:100%;background:${getScoreColor(match.similarity)};border-radius:4px;transition:width 0.6s ease"></div>
            </div>
            <div class="match-similarity" style="color:${getScoreColor(match.similarity)};min-width:48px;text-align:right">
                ${match.similarity}%
            </div>
            <div class="match-confidence" style="color:${confidenceColor};font-size:0.75rem;min-width:52px;text-align:right">
                ${confidenceLabel}
            </div>`;
        matchesList.appendChild(item);
    });
}

// ─── Analyze Recitation ─────────────────────────────────────────────────────────

const analyzeUpload = setupFileUpload('analyze');

let analyzeRadarChart = null;

// Enable submit button when file is loaded and surah is selected
function checkAnalyzeReady() {
    const file = analyzeUpload.getFile();
    const surah = document.getElementById('surah-select').value;
    document.getElementById('analyze-submit-btn').disabled = !(file && surah);
}

document.getElementById('surah-select').addEventListener('change', checkAnalyzeReady);

document.getElementById('analyze-submit-btn').addEventListener('click', async () => {
    const file = analyzeUpload.getFile();
    if (!file) { showToast('Please select an audio file.', 'error'); return; }

    const surahName = document.getElementById('surah-select').value;
    const referenceQari = document.getElementById('qari-select').value;

    if (!surahName) { showToast('Please select a Surah.', 'error'); return; }

    const formData = new FormData();
    formData.append('audio_file', file);
    formData.append('surah_name', surahName);
    if (referenceQari) formData.append('reference_qari', referenceQari);

    const steps = [
        'Loading reference audio…',
        'Checking timing alignment…',
        'Analyzing melody contour…',
        'Detecting breath patterns…',
        'Generating report…'
    ];
    let stepIdx = 0;
    setLoading('analyze', true, steps[0]);
    const stepInterval = setInterval(() => {
        stepIdx = (stepIdx + 1) % steps.length;
        const stepEl = document.getElementById('analyze-loading-step');
        if (stepEl) stepEl.textContent = steps[stepIdx];
    }, 2500);

    document.getElementById('analyze-results').style.display = 'none';

    try {
        const res = await fetch(`${API_BASE}/api/analyze-recitation`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Analysis failed');
        }

        const data = await res.json();
        renderAnalyzeResults(data.report);
        showToast('Analysis complete!', 'success');

    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    } finally {
        clearInterval(stepInterval);
        setLoading('analyze', false);
    }
});

function renderAnalyzeResults(report) {
    const resultsEl = document.getElementById('analyze-results');
    resultsEl.style.display = 'block';
    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });

    const overall = report.overall_score;
    const timing = report.timing.timing_score;
    const melody = report.melody.melody_score;
    const breath = report.breath.breath_score;

    // Overall circle animation
    document.getElementById('overall-score-value').textContent = overall;
    document.getElementById('overall-interpretation').textContent = report.overall_interpretation;
    document.getElementById('analysis-reference-info').textContent =
        `Compared to: ${report.reference_qari} · ${report.surah.replace(/_/g, ' ')}`;

    const circumference = 327;
    const offset = circumference - (overall / 100) * circumference;
    const circle = document.getElementById('score-circle-fill');
    circle.style.stroke = getScoreColor(overall);
    setTimeout(() => { circle.style.strokeDashoffset = offset; }, 100);

    // Timing card
    document.getElementById('timing-score').textContent = formatScore(timing);
    document.getElementById('timing-interpretation').textContent = report.timing.interpretation;
    const timingBar = document.getElementById('timing-bar');
    setTimeout(() => {
        timingBar.style.width = `${timing}%`;
        timingBar.style.background = getScoreColor(timing);
    }, 100);

    // Melody card
    document.getElementById('melody-score').textContent = formatScore(melody);
    document.getElementById('melody-interpretation').textContent = report.melody.interpretation;
    const melodyBar = document.getElementById('melody-bar');
    setTimeout(() => {
        melodyBar.style.width = `${melody}%`;
        melodyBar.style.background = getScoreColor(melody);
    }, 200);

    // Breath card
    document.getElementById('breath-score').textContent = formatScore(breath);
    document.getElementById('breath-interpretation').textContent = report.breath.interpretation;
    const breathBar = document.getElementById('breath-bar');
    setTimeout(() => {
        breathBar.style.width = `${breath}%`;
        breathBar.style.background = getScoreColor(breath);
    }, 300);

    // Breath detail boxes
    const detailsGrid = document.getElementById('breath-details');
    detailsGrid.innerHTML = `
        <div class="detail-item">
            <strong>${report.breath.user_pause_count}</strong>
            <span>Your Breath Pauses</span>
        </div>
        <div class="detail-item">
            <strong>${report.breath.reference_pause_count}</strong>
            <span>Reference Pauses</span>
        </div>
        <div class="detail-item">
            <strong>${report.breath.user_avg_pause_duration}s</strong>
            <span>Your Avg Pause Duration</span>
        </div>
        <div class="detail-item">
            <strong>${report.breath.reference_avg_pause_duration}s</strong>
            <span>Reference Avg Duration</span>
        </div>`;

    // Radar chart
    if (analyzeRadarChart) analyzeRadarChart.destroy();
    analyzeRadarChart = new Chart(document.getElementById('analyze-radar-chart'), {
        type: 'radar',
        data: {
            labels: ['Timing', 'Melody', 'Breathing'],
            datasets: [
                {
                    label: 'Your Recitation',
                    data: [timing, melody, breath],
                    backgroundColor: 'rgba(37, 99, 235, 0.2)',
                    borderColor: '#2563eb',
                    borderWidth: 3,
                    pointBackgroundColor: '#2563eb'
                },
                {
                    label: 'Perfect Score',
                    data: [100, 100, 100],
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderColor: '#10b981',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointBackgroundColor: '#10b981'
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    min: 0, max: 100,
                    ticks: { stepSize: 20 }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Recitation Quality Radar',
                    font: { size: 16, weight: 'bold' }
                },
                legend: { position: 'bottom' }
            }
        }
    });
}

// ─── Compare All Qaris ─────────────────────────────────────────────────────────

const compareUpload = setupFileUpload('compare');

let compareBarChart = null;

document.getElementById('compare-submit-btn').addEventListener('click', async () => {
    const file = compareUpload.getFile();
    if (!file) { showToast('Please select an audio file.', 'error'); return; }

    const formData = new FormData();
    formData.append('audio_file', file);

    setLoading('compare', true);
    document.getElementById('compare-results').style.display = 'none';

    try {
        const res = await fetch(`${API_BASE}/api/compare-all-qaris`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Comparison failed');
        }

        const data = await res.json();
        renderCompareResults(data);
        showToast('Comparison complete!', 'success');

    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    } finally {
        setLoading('compare', false);
    }
});

function renderCompareResults(data) {
    const resultsEl = document.getElementById('compare-results');
    resultsEl.style.display = 'block';
    resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Stats row
    const statsRow = document.getElementById('compare-stats-row');
    statsRow.innerHTML = `
        <div class="stat-card">
            <h4>Best Match</h4>
            <p style="font-size:1.2rem">${data.best_match.qari}</p>
        </div>
        <div class="stat-card">
            <h4>Highest Similarity</h4>
            <p>${data.statistics.highest_similarity}%</p>
        </div>
        <div class="stat-card">
            <h4>Average Similarity</h4>
            <p>${data.statistics.average_similarity}%</p>
        </div>
        <div class="stat-card">
            <h4>Qaris Compared</h4>
            <p>${data.total_qaris_compared}</p>
        </div>`;

    // Horizontal bar chart
    const comparisons = data.all_comparisons;
    const labels = comparisons.map(c => c.qari);
    const values = comparisons.map(c => c.similarity_percent);
    const colors = values.map(v => getScoreColor(v) + 'cc');

    if (compareBarChart) compareBarChart.destroy();
    compareBarChart = new Chart(document.getElementById('compare-bar-chart'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Similarity (%)',
                data: values,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('cc', '')),
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Full Comparison: Your Voice vs All 12 Qaris',
                    font: { size: 16, weight: 'bold' }
                }
            },
            scales: {
                x: { min: 0, max: 100, title: { display: true, text: 'Similarity (%)' } }
            }
        }
    });

    // Table
    const tbody = document.getElementById('compare-table-body');
    tbody.innerHTML = '';
    comparisons.forEach((c, i) => {
        const levelClass = c.match_level.toLowerCase().split(' ')[0];
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>#${i + 1}</td>
            <td><strong>${c.qari}</strong></td>
            <td>
                <div style="display:flex;align-items:center;gap:1rem">
                    <div style="flex:1;background:#e2e8f0;height:8px;border-radius:4px">
                        <div style="width:${c.similarity_percent}%;height:100%;background:${getScoreColor(c.similarity_percent)};border-radius:4px"></div>
                    </div>
                    <strong>${c.similarity_percent}%</strong>
                </div>
            </td>
            <td><span class="match-level-badge ${levelClass}">${c.match_level}</span></td>`;
        tbody.appendChild(row);
    });
}

// ─── Initialize on Page Load ────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    await loadDropdowns();

    // Override analyze submit button default disabled — it should wait for file AND surah
    document.getElementById('analyze-submit-btn').disabled = true;
});
