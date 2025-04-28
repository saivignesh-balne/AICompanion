document.addEventListener('DOMContentLoaded', function () {
    let currentAlarm = null;

    function checkAlarms() {
        fetch('/check_alarm', { credentials: 'include' })
            .then(response => response.json())
            .then(data => {
                if (data.alarm_triggered && (!currentAlarm || currentAlarm.type !== data.type)) {
                    triggerAlarm(data);
                }
            })
            .catch(error => console.error('Error checking alarms:', error));
    }

    function triggerAlarm(alarmData) {
        // Stop any existing alarm
        if (currentAlarm) {
            currentAlarm.audio.pause();
            currentAlarm.audio.currentTime = 0;
        }

        // Create new alarm
        currentAlarm = {
            type: alarmData.type,
            audio: new Audio(alarmData.audio_file),
            message: alarmData.message
        };

        // Configure audio
        currentAlarm.audio.loop = true;

        // Update UI
        const alarmContainer = document.getElementById('alarm-container');
        const alarmType = document.getElementById('alarm-type');
        const alarmAudio = document.getElementById('alarm-audio');

        if (alarmContainer && alarmType && alarmAudio) {
            alarmType.textContent = alarmData.type + ' Reminder';
            alarmAudio.src = alarmData.audio_file;

            // Play the audio programmatically
            currentAlarm.audio.play().catch(error => console.error('Audio play failed:', error));
            alarmContainer.style.display = 'block';
        }
    }

    window.stopAlarm = function () {
        if (currentAlarm && currentAlarm.audio) {
            // Stop and reset the audio
            currentAlarm.audio.pause();
            currentAlarm.audio.currentTime = 0;

            // Also stop the audio element in the UI
            const alarmAudio = document.getElementById('alarm-audio');
            if (alarmAudio) {
                alarmAudio.pause();
                alarmAudio.currentTime = 0;
            }

            currentAlarm = null; // Clear the current alarm reference
        }

        // Hide the alarm container
        const alarmContainer = document.getElementById('alarm-container');
        if (alarmContainer) {
            alarmContainer.style.display = 'none';
        }
    };

    // Check alarms every 1 second
    setInterval(checkAlarms, 1000);
});
