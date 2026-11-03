/**
 * Traffic-Vigil AI - Frontend Controller
 * Handles Chart.js visualization, evidence modals, and reviewer updates.
 */

// ================= Evidence Modal Handling =================
function openEvidenceModal(imageSrc, title, time, event, vehicle, plate, conf, status, notes) {
    const modal = document.getElementById('evidenceModal');
    if (!modal) return;

    document.getElementById('modalEvidenceImage').src = imageSrc;
    document.getElementById('modalTitle').textContent = title || 'Evidence Inspection';
    document.getElementById('modalMeta').textContent = `Timestamp: ${time} | Plate: ${plate}`;
    document.getElementById('modalEvent').textContent = event;
    document.getElementById('modalVehicle').textContent = vehicle ? vehicle.toUpperCase() : '-';
    document.getElementById('modalPlate').textContent = plate || 'N/A';
    document.getElementById('modalConfidence').textContent = conf;
    document.getElementById('modalStatusBadge').textContent = status;
    document.getElementById('modalNotes').textContent = notes || 'No reviewer notes logged.';
    document.getElementById('modalDownloadBtn').href = imageSrc;

    modal.classList.add('active');
}

function closeEvidenceModal() {
    const modal = document.getElementById('evidenceModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

// ================= Review Modal & AJAX Handling =================
function openReviewModal(recordId, currentStatus, currentNotes) {
    const modal = document.getElementById('reviewModal');
    if (!modal) return;

    document.getElementById('reviewRecordId').textContent = recordId;
    document.getElementById('reviewViolationId').value = recordId;
    
    const select = document.getElementById('reviewStatusSelect');
    if (select) {
        select.value = currentStatus || 'Approved Violation';
    }

    const notesInput = document.getElementById('reviewNotesInput');
    if (notesInput) {
        notesInput.value = (currentNotes && currentNotes !== 'None' && currentNotes !== 'No notes') ? currentNotes : '';
    }

    modal.classList.add('active');
}

function closeReviewModal() {
    const modal = document.getElementById('reviewModal');
    if (modal) {
        modal.classList.remove('active');
    }
}

function submitReview(event) {
    event.preventDefault();
    const violationId = document.getElementById('reviewViolationId').value;
    const newStatus = document.getElementById('reviewStatusSelect').value;
    const notes = document.getElementById('reviewNotesInput').value;

    fetch('/api/update_review', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            violation_id: parseInt(violationId),
            review_status: newStatus,
            notes: notes
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            updateBadgeUI(violationId, newStatus);
            closeReviewModal();
            showToast('Review decision recorded successfully.', 'success');
        } else {
            alert('Failed to update review status: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(err => {
        console.error('Error submitting review:', err);
        alert('Server communication error. Check console.');
    });
}

function quickUpdateReview(violationId, status) {
    fetch('/api/update_review', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            violation_id: parseInt(violationId),
            review_status: status,
            notes: `Quick decision marked: ${status}`
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            updateBadgeUI(violationId, status);
            showToast(`Record #${violationId} marked as "${status}".`, 'success');
        } else {
            alert('Failed to update status.');
        }
    })
    .catch(err => {
        console.error('Error updating review:', err);
    });
}

function updateBadgeUI(violationId, newStatus) {
    const badge = document.getElementById(`status-badge-${violationId}`);
    if (badge) {
        badge.textContent = newStatus;
        // reset classes
        badge.className = 'review-status-pill';
        const slug = newStatus.toLowerCase().replace(/[\s\/]+/g, '-');
        badge.classList.add(`status-${slug}`);
    }
}

// Toast helper
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `alert alert-${type}`;
    toast.style.position = 'fixed';
    toast.style.bottom = '20px';
    toast.style.right = '20px';
    toast.style.zIndex = '9999';
    toast.style.minWidth = '280px';
    toast.innerHTML = `
        <div class="alert-icon"><i class="fa-solid fa-circle-check"></i></div>
        <div class="alert-message">${message}</div>
    `;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.4s ease';
        setTimeout(() => toast.remove(), 400);
    }, 3000);
}

// Close modals when clicking backdrop
window.addEventListener('click', function (e) {
    const evidenceModal = document.getElementById('evidenceModal');
    const reviewModal = document.getElementById('reviewModal');
    if (e.target === evidenceModal) closeEvidenceModal();
    if (e.target === reviewModal) closeReviewModal();
});

// Close modals with Escape key
window.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
        closeEvidenceModal();
        closeReviewModal();
    }
});

// ================= Dashboard Chart.js Visualizations =================
function renderDashboardCharts(stats) {
    const vDistCanvas = document.getElementById('vehicleDistributionChart');
    const eDistCanvas = document.getElementById('eventDistributionChart');

    if (vDistCanvas) {
        const vData = stats.vehicle_distribution || {};
        const labels = Object.keys(vData).map(k => k.charAt(0).toUpperCase() + k.slice(1));
        const counts = Object.values(vData);

        const hasData = counts.length > 0 && counts.some(c => c > 0);

        new Chart(vDistCanvas, {
            type: 'doughnut',
            data: {
                labels: hasData ? labels : ['No Vehicles Yet'],
                datasets: [{
                    data: hasData ? counts : [1],
                    backgroundColor: hasData ? [
                        '#3b82f6', // car (blue)
                        '#f59e0b', // motorcycle (amber)
                        '#10b981', // bus (emerald)
                        '#a855f7'  // truck (purple)
                    ] : ['#e2e8f0'],
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: { family: 'Plus Jakarta Sans', size: 12 },
                            usePointStyle: true
                        }
                    }
                },
                cutout: '68%'
            }
        });
    }

    if (eDistCanvas) {
        const eData = stats.event_distribution || {};
        const labels = Object.keys(eData);
        const counts = Object.values(eData);

        const hasData = counts.length > 0 && counts.some(c => c > 0);

        new Chart(eDistCanvas, {
            type: 'bar',
            data: {
                labels: hasData ? labels : ['No Events Logged'],
                datasets: [{
                    label: 'Occurrences',
                    data: hasData ? counts : [0],
                    backgroundColor: '#2563eb',
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { stepSize: 1, precision: 0 }
                    },
                    x: {
                        ticks: {
                            font: { family: 'Plus Jakarta Sans', size: 11 },
                            maxRotation: 20
                        }
                    }
                }
            }
        });
    }
}
