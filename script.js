let historicalData = [];
let emergencyMode = false;

function fetchTrafficData() {
    fetch('/traffic-analysis')
        .then(response => response.json())
        .then(data => {
            updateDashboard(data);
            updateCharts(data);
            checkEmergency(data);
        })
        .catch(error => console.error("Error fetching data:", error));
}

function fetchHistoricalData() {
    fetch('/historical-data')
        .then(response => response.json())
        .then(data => {
            historicalData = data;
            renderHistoricalChart();
        })
        .catch(error => console.error("Error fetching historical data:", error));
}

function updateDashboard(data) {
    document.getElementById('speed').textContent = data.vehicle_speed;
    document.getElementById('density').textContent = `${data.traffic_density} (${data.congestion_level})`;
    document.getElementById('road_traffic').textContent = data.further_road_traffic ? 'High' : 'Low';
    document.getElementById('signal_time').textContent = data.optimized_signal_time;
    document.getElementById('emergency').textContent = data.emergency_vehicle ? '⚠️ Detected' : 'None';
    document.getElementById('pedestrians').textContent = data.pedestrian_crossing ? 'Waiting' : 'None';
    document.getElementById('prediction').textContent = Math.round(data.predicted_density || 0);
    
    // Update timestamp
    document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
    
    // Visual feedback for emergency
    if (data.emergency_vehicle) {
        document.body.classList.add('emergency-mode');
        emergencyMode = true;
    } else if (emergencyMode) {
        document.body.classList.remove('emergency-mode');
        emergencyMode = false;
    }
}

function updateCharts(data) {
    // Update real-time gauge charts
    updateGauge('speed-gauge', data.vehicle_speed, 0, 120);
    updateGauge('density-gauge', data.traffic_density, 0, 150);
    updateGauge('signal-gauge', data.optimized_signal_time, 10, 120);
}

function updateGauge(elementId, value, min, max) {
    const element = document.getElementById(elementId);
    if (element) {
        const percentage = ((value - min) / (max - min)) * 100;
        element.style.background = `conic-gradient(
            #00FF7F ${percentage}%, 
            rgba(255, 255, 255, 0.2) ${percentage}% 100%
        )`;
        element.setAttribute('data-value', value);
    }
}

function checkEmergency(data) {
    if (data.emergency_vehicle) {
        // Flash notification
        const notification = document.createElement('div');
        notification.className = 'emergency-notification';
        notification.innerHTML = '🚑 EMERGENCY VEHICLE DETECTED! PRIORITY ROUTING ACTIVE!';
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}

function renderHistoricalChart() {
    if (historicalData.length === 0) return;
    
    const ctx = document.getElementById('history-chart').getContext('2d');
    
    // Process data for chart
    const labels = historicalData.map(entry => 
        new Date(entry.timestamp).toLocaleTimeString());
    const densities = historicalData.map(entry => entry.traffic_density);
    const speeds = historicalData.map(entry => entry.vehicle_speed);
    
    if (window.historyChart) {
        window.historyChart.destroy();
    }
    
    window.historyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Traffic Density',
                    data: densities,
                    borderColor: '#FFD700',
                    backgroundColor: 'rgba(255, 215, 0, 0.1)',
                    tension: 0.3,
                    yAxisID: 'y'
                },
                {
                    label: 'Vehicle Speed (km/h)',
                    data: speeds,
                    borderColor: '#00FF7F',
                    backgroundColor: 'rgba(0, 255, 127, 0.1)',
                    tension: 0.3,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Density'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Speed (km/h)'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

// Initialize the dashboard
function initDashboard() {
    fetchTrafficData();
    fetchHistoricalData();
    setInterval(fetchTrafficData, 3000);
    setInterval(fetchHistoricalData, 30000);
    
    // Create gauge elements
    createGauge('speed-gauge', 'Speed (km/h)');
    createGauge('density-gauge', 'Density');
    createGauge('signal-gauge', 'Signal Time (s)');
}

function createGauge(id, label) {
    const container = document.getElementById('gauges-container');
    const gauge = document.createElement('div');
    gauge.className = 'gauge';
    gauge.id = id;
    
    const labelElement = document.createElement('div');
    labelElement.className = 'gauge-label';
    labelElement.textContent = label;
    
    const valueElement = document.createElement('div');
    valueElement.className = 'gauge-value';
    valueElement.id = `${id}-value`;
    
    gauge.appendChild(labelElement);
    gauge.appendChild(valueElement);
    container.appendChild(gauge);
}

document.addEventListener('DOMContentLoaded', initDashboard);