// Main JavaScript file for custom interactions

// Namespace for charting functions to avoid global scope pollution
const AppCharts = {
    colors: [
        'rgba(54, 162, 235, 0.7)', // Blue
        'rgba(255, 99, 132, 0.7)', // Red
        'rgba(75, 192, 192, 0.7)', // Green
        'rgba(255, 206, 86, 0.7)', // Yellow
        'rgba(153, 102, 255, 0.7)', // Purple
        'rgba(255, 159, 64, 0.7)', // Orange
        'rgba(100, 100, 100, 0.7)', // Grey
        'rgba(0, 200, 220, 0.7)' // Cyan
    ],
    borderColors: [
        'rgba(54, 162, 235, 1)',
        'rgba(255, 99, 132, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(153, 102, 255, 1)',
        'rgba(255, 159, 64, 1)',
        'rgba(100, 100, 100, 1)',
        'rgba(0, 200, 220, 1)'
    ],
    currentCharts: {}, // To keep track of chart instances for updates/destruction

    destroyChart: function(chartId) {
        if (this.currentCharts[chartId]) {
            this.currentCharts[chartId].destroy();
            delete this.currentCharts[chartId];
        }
    },

    renderClusterDistributionChart: function(visualsData) {
        this.destroyChart('clusterDistributionChart');
        const ctx = document.getElementById('clusterDistributionChart')?.getContext('2d');
        if (!ctx || !visualsData || !visualsData.labels || !visualsData.data) {
            // console.warn('Cluster distribution chart: Missing canvas or data.');
            return;
        }
        this.currentCharts['clusterDistributionChart'] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: visualsData.labels,
                datasets: [{
                    label: 'Number of Customers per Cluster',
                    data: visualsData.data,
                    backgroundColor: this.colors.slice(0, visualsData.data.length),
                    borderColor: this.borderColors.slice(0, visualsData.data.length),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Number of Customers' }
                    },
                    x: {
                        title: { display: true, text: 'Cluster ID' }
                    }
                },
                plugins: {
                    legend: {
                        display: false // Or true if you want it
                    }
                }
            }
        });
    },

    renderScatterPlot: function(scatterData, featureX = 'Feature X', featureY = 'Feature Y') {
        this.destroyChart('clusterScatterPlot');
        const ctx = document.getElementById('clusterScatterPlot')?.getContext('2d');
         if (!ctx || !scatterData || !scatterData.datasets || scatterData.datasets.length === 0) {
            // console.warn('Scatter plot: Missing canvas or data.');
            // Optionally, display a message in the canvas container
            const container = document.getElementById('scatterPlotContainer');
            if (container) container.innerHTML = '<p class="text-muted text-center small">Scatter plot data not available or insufficient.</p>';
            return;
        }

        // Ensure datasets have colors
        scatterData.datasets.forEach((dataset, index) => {
            dataset.backgroundColor = dataset.backgroundColor || this.colors[index % this.colors.length];
            dataset.borderColor = dataset.borderColor || this.borderColors[index % this.borderColors.length];
            dataset.pointRadius = dataset.pointRadius || 5;
            dataset.pointHoverRadius = dataset.pointHoverRadius || 7;
        });

        this.currentCharts['clusterScatterPlot'] = new Chart(ctx, {
            type: 'scatter',
            data: scatterData, // Expects { datasets: [{ label: 'Cluster 0', data: [{x: val, y: val}, ...]}, ...] }
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: { display: true, text: featureX }
                    },
                    y: {
                        title: { display: true, text: featureY }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: (${context.raw.x}, ${context.raw.y})`;
                            }
                        }
                    }
                }
            }
        });
    },

    renderRadarChart: function(radarData) { // { labels: ['F1', 'F2', ...], datasets: [{label: 'C1', data:[]}, {label: 'C2', data:[]}]}
        this.destroyChart('clusterRadarChart');
        const ctx = document.getElementById('clusterRadarChart')?.getContext('2d');
        if (!ctx || !radarData || !radarData.labels || !radarData.datasets || radarData.datasets.length === 0) {
            // console.warn('Radar chart: Missing canvas or data.');
            const container = document.getElementById('radarChartContainer');
            if (container) container.innerHTML = '<p class="text-muted text-center small">Radar chart data not available or insufficient.</p>';
            return;
        }

        // Ensure datasets have colors
        radarData.datasets.forEach((dataset, index) => {
            dataset.backgroundColor = dataset.backgroundColor || this.colors[index % this.colors.length].replace('0.7', '0.2'); // Lighter for area
            dataset.borderColor = dataset.borderColor || this.borderColors[index % this.borderColors.length];
            dataset.pointBackgroundColor = dataset.pointBackgroundColor || this.borderColors[index % this.borderColors.length];
            dataset.borderWidth = dataset.borderWidth || 2;
        });

        this.currentCharts['clusterRadarChart'] = new Chart(ctx, {
            type: 'radar',
            data: radarData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                elements: {
                    line: {
                        borderWidth: 2
                    }
                },
                scales: {
                    r: {
                        angleLines: { display: true },
                        suggestedMin: 0, // Or adjust based on data (e.g. if normalized)
                        // suggestedMax: 1, // If data is normalized between 0 and 1
                        pointLabels: {
                            font: {
                                size: 10
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    }
                }
            }
        });
    }
};


document.addEventListener('DOMContentLoaded', function () {
    // Update file input label
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            var fileName = e.target.files[0] ? e.target.files[0].name : 'Select file...';
            var nextSibling = e.target.nextElementSibling;
            if (nextSibling && nextSibling.classList.contains('custom-file-label')) {
                nextSibling.innerText = fileName;
            }
        });
    }

    // Handle form submission spinner and client-side validation
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const uploadFeedback = document.getElementById('upload-feedback');

    if (uploadForm && fileInput && uploadFeedback) {
        uploadForm.addEventListener('submit', function(event) {
            uploadFeedback.innerHTML = ''; // Clear previous feedback
            const file = fileInput.files[0];
            let isValid = true;

            // Rule 1: File must be selected
            if (!file) {
                displayValidationFeedback('Please select a file.', 'danger');
                event.preventDefault();
                isValid = false;
                return; // Stop further processing if no file
            }

            // Rule 2: File must be a CSV
            if (!file.name.toLowerCase().endsWith('.csv')) {
                displayValidationFeedback('Invalid file type. Please upload a CSV file.', 'danger');
                event.preventDefault();
                isValid = false;
            }

            // Rule 3: File size (e.g., max 5MB) - Optional
            const maxSizeMB = 5;
            if (file.size > maxSizeMB * 1024 * 1024) {
                displayValidationFeedback(`File is too large. Maximum size is ${maxSizeMB}MB.`, 'danger');
                event.preventDefault();
                isValid = false;
            }

            if (!isValid) {
                return; // Stop if any validation failed
            }

            event.preventDefault(); // Prevent default form submission for AJAX

            const submitButton = uploadForm.querySelector('button[type="submit"]');
            const spinner = submitButton.querySelector('.spinner-border');

            if (spinner) spinner.classList.remove('d-none');
            submitButton.disabled = true;
            uploadFeedback.innerHTML = '<div class="alert alert-info">Processing your file... Please wait.</div>';

            const formData = new FormData(uploadForm);
            const uploadUrl = uploadForm.dataset.uploadUrl; // Get URL from data attribute

            fetch(uploadUrl, {
                method: 'POST',
                body: formData,
            })
            .then(response => {
                // Check if response is JSON, otherwise handle as error
                const contentType = response.headers.get("content-type");
                if (contentType && contentType.indexOf("application/json") !== -1) {
                    return response.json();
                } else {
                    return response.text().then(text => { throw new Error("Server returned non-JSON response: " + text) });
                }
            })
            .then(data => {
                if (spinner) spinner.classList.add('d-none');
                submitButton.disabled = false;

                if (data.success) {
                    displayValidationFeedback(data.message || 'File processed successfully!', 'success');
                    updatePageWithResults(data);
                } else {
                    displayValidationFeedback(data.error || 'An unknown error occurred.', 'danger');
                    clearResultsOnError(); // Clear old results if new submission fails
                }
            })
            .catch(error => {
                if (spinner) spinner.classList.add('d-none');
                submitButton.disabled = false;
                displayValidationFeedback(`Request failed: ${error.message}`, 'danger');
                console.error('Error during AJAX upload:', error);
                clearResultsOnError();
            });
        });
    }

    // Update the fetch URL to be dynamic, passed from the template or a global JS var
    // This is a placeholder. In a real setup, the URL would be passed via a data attribute or global JS var.
    // For now, I'll hardcode it, assuming the JS is loaded in a way that this relative path works.
    // Ideally, the Flask URL should be provided to the script.
    // For example, in index.html: <script>const uploadUrl = "{{ url_for('upload_file') }}";</script>
    // And then use `uploadUrl` in fetch. For now, I'll use a relative path.
    // This will be corrected if it causes issues during testing.
    // const UPLOAD_URL = '/upload'; // Replaced the Jinja templating.

function updatePageWithResults(data) {
    const insightsContainer = document.getElementById('insights-section-container');
    if (insightsContainer && data.insights_section_html) {
        insightsContainer.innerHTML = data.insights_section_html;
    } else if (insightsContainer) {
        insightsContainer.innerHTML = '<p class="text-muted">Insights not available.</p>'; // Fallback
    }

    const resultsContainer = document.getElementById('results-section-container');
    if (resultsContainer && data.results_section_html) {
        resultsContainer.innerHTML = data.results_section_html;
    } else if (resultsContainer) {
        resultsContainer.innerHTML = '<p class="text-muted">Results not available.</p>'; // Fallback
    }

    const vizContainer = document.getElementById('visualizations-section-container');
    if (vizContainer && data.visualizations_section_html) {
        vizContainer.innerHTML = data.visualizations_section_html;
        // Charts need to be re-initialized AFTER their canvas elements are in the DOM
        if (data.visuals_data_json) {
            if (data.visuals_data_json.cluster_distribution && Object.keys(data.visuals_data_json.cluster_distribution).length > 0) {
                AppCharts.renderClusterDistributionChart(data.visuals_data_json.cluster_distribution);
            } else {
                AppCharts.destroyChart('clusterDistributionChart');
            }
            // TODO: Add similar for scatter and radar when data is available
            if (data.visuals_data_json.scatter_plot && Object.keys(data.visuals_data_json.scatter_plot).length > 0 && data.visuals_data_json.scatter_plot.data.datasets.length > 0) {
                AppCharts.renderScatterPlot(data.visuals_data_json.scatter_plot.data, data.visuals_data_json.scatter_plot.feature_x, data.visuals_data_json.scatter_plot.feature_y);
            } else {
                AppCharts.destroyChart('clusterScatterPlot');
                 // Optionally update the specific canvas container to say "data not available"
                const scatterContainer = document.getElementById('scatterPlotContainer');
                if(scatterContainer) scatterContainer.innerHTML = '<p class="text-muted text-center small">Scatter plot data not available or insufficient for selected features.</p>';
            }
            if (data.visuals_data_json.radar_chart && Object.keys(data.visuals_data_json.radar_chart).length > 0 && data.visuals_data_json.radar_chart.datasets.length > 0) {
                AppCharts.renderRadarChart(data.visuals_data_json.radar_chart);
            } else {
                AppCharts.destroyChart('clusterRadarChart');
                const radarContainer = document.getElementById('radarChartContainer');
                if(radarContainer) radarContainer.innerHTML = '<p class="text-muted text-center small">Radar chart data not available or insufficient.</p>';
            }
        }
    } else if (vizContainer) {
        vizContainer.innerHTML = '<p class="text-muted">Visualizations not available.</p>'; // Fallback
        AppCharts.destroyChart('clusterDistributionChart');
        AppCharts.destroyChart('clusterScatterPlot');
        AppCharts.destroyChart('clusterRadarChart');
    }

    const recContainer = document.getElementById('recommendations-section-container');
    if (recContainer && data.recommendations_section_html) {
        recContainer.innerHTML = data.recommendations_section_html;
    } else if (recContainer) {
        recContainer.innerHTML = '<p class="text-muted">Recommendations not available.</p>'; // Fallback
    }
}

function clearResultsOnError() {
    // Hide or clear all sections that show results
    const sectionsToClear = [
        'results-section-container',
        'insights-section-container',
        'visualizations-section-container',
        'recommendations-section-container'
    ];
    sectionsToClear.forEach(id => {
        const section = document.getElementById(id);
        if (section) {
            // Option 1: Just hide
            section.style.display = 'none';
            // Option 2: Clear content (example for results table)
            if (id === 'results-section') {
                 const container = section.querySelector('#results-table-container');
                 if (container) container.innerHTML = '<p class="text-muted">Previous results cleared due to error.</p>';
            }
             if (id === 'insights-section') {
                const insightsContent = section.querySelector('.card-body'); // Example
                if(insightsContent) insightsContent.innerHTML = '<p class="text-muted">Previous insights cleared.</p>';
            }
        }
    });
    AppCharts.destroyChart('clusterDistributionChart');
    AppCharts.destroyChart('clusterScatterPlot');
    AppCharts.destroyChart('clusterRadarChart');
}

    // Initial chart rendering if data is embedded in the page
    // (e.g. when page reloads after a non-AJAX form submission)
    if (typeof initialVisualsData !== 'undefined' && initialVisualsData && initialVisualsData.cluster_distribution) {
        AppCharts.renderClusterDistributionChart(initialVisualsData.cluster_distribution);
    }
    if (typeof initialScatterData !== 'undefined' && initialScatterData) {
        AppCharts.renderScatterPlot(initialScatterData.data, initialScatterData.feature_x, initialScatterData.feature_y);
    }
    if (typeof initialRadarData !== 'undefined' && initialRadarData) {
        AppCharts.renderRadarChart(initialRadarData);
    }
});

// Expose chart rendering functions if they need to be called by inline scripts from Flask templates
// (e.g. if Flask passes data directly to a script tag that then calls these)
// window.renderClusterDistributionChart = AppCharts.renderClusterDistributionChart.bind(AppCharts);
// window.renderScatterPlot = AppCharts.renderScatterPlot.bind(AppCharts);
// window.renderRadarChart = AppCharts.renderRadarChart.bind(AppCharts);

// Note: The `initialVisualsData`, `initialScatterData`, `initialRadarData` variables
// are expected to be defined globally (e.g. in a script tag in index.html) if charts
// need to be rendered on page load using data from Flask.
// Example in index.html (or _base.html):
// <script>
//     const initialVisualsData = {{ visuals_data_json | default('{}') | tojson | safe }};
//     const initialScatterData = {{ scatter_data_json | default('{}') | tojson | safe }};
//     const initialRadarData = {{ radar_data_json | default('{}') | tojson | safe }};
// </script>
// Then app.py needs to pass 'visuals_data_json', 'scatter_data_json', 'radar_data_json' to render_template.
// The 'cluster_distribution' key within 'visuals_data_json' is specific to how it was structured before.
// It might be simpler to pass each chart's data object directly.
// E.g., cluster_dist_data = {'labels': [...], 'data': [...]}
// Then in JS: AppCharts.renderClusterDistributionChart(cluster_dist_data);
// This structure is used in the `DOMContentLoaded` listener.

// For AJAX updates, these functions would be called from the success handler of the fetch request.
// The AJAX handler itself is planned for a later step.
console.log("main.js loaded");

function displayValidationFeedback(message, type) {
    const feedbackDiv = document.getElementById('upload-feedback');
    if (feedbackDiv) {
        feedbackDiv.innerHTML = `<div class="alert alert-${type} alert-dismissible fade show" role="alert">
                                ${message}
                                <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                                    <span aria-hidden="true">&times;</span>
                                </button>
                            </div>`;
    }
    // Also, add/remove Bootstrap's is-invalid class for more specific input feedback if desired
    // const fileInputEl = document.getElementById('fileInput');
    // if (type === 'danger') {
    //     fileInputEl.classList.add('is-invalid');
    // } else {
    //     fileInputEl.classList.remove('is-invalid');
    // }
}
