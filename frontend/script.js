// Flappy Bird Emotion Game - JavaScript
// Emotion-based Dynamic Difficulty Adjustment (DDA)

(function() {
    'use strict';

    // Game Configuration
    const GAME_CONFIG = {
        FPS: 60,
        CANVAS_WIDTH: 800,
        CANVAS_HEIGHT: 600,
        GRAVITY: 0.35,  // Reduced from 0.5
        FLAP_POWER: -7,  // Reduced from -8
        PIPE_WIDTH: 80,
        PIPE_GAP: 200,  // Increased gap for easier passage
        PIPE_SPEED: 0.8,  // Much slower pipes only
        PIPE_SPACING: 350,  // Increased from 300
        BIRD_SIZE: 30,
        BIRD_X: 150
    };

    // Game State
    let gameState = {
        isRunning: false,
        isPaused: false,
        isGameOver: false,
        score: 0,
        sessionId: null,
        gameStartTime: null,
        lastEmotionCheck: 0,
        emotionCheckInterval: 3500,
        currentEmotion: null,
        difficultyLevel: 1.0,
        maxDifficulty: 1.0,
        emotionsDetected: 0,
        // Flappy Bird specific difficulty parameters
        currentGravity: GAME_CONFIG.GRAVITY,
        currentFlapPower: GAME_CONFIG.FLAP_POWER,
        currentPipeGap: GAME_CONFIG.PIPE_GAP,
        currentPipeSpeed: GAME_CONFIG.PIPE_SPEED,
        currentPipeSpacing: GAME_CONFIG.PIPE_SPACING
    };

    // Bird object
    let bird = {
        x: GAME_CONFIG.BIRD_X,
        y: GAME_CONFIG.CANVAS_HEIGHT / 2,
        velocity: 0,
        size: GAME_CONFIG.BIRD_SIZE
    };

    // Pipes array
    let pipes = [];
    let gameLoop;
    let emotionDetectionInterval;
    let lastTime = 0;

    // DOM Elements
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    const video = document.getElementById('video');
    const emotionDisplay = document.getElementById('emotion');
    const scoreDisplay = document.getElementById('score');
    const difficultyDisplay = document.getElementById('difficulty');
    const statsSection = document.getElementById('statsSection');
    const modelStatusDisplay = document.getElementById('modelStatus');

    // Backend API Configuration
    const BACKEND_URL = '/api';  // Use relative path for backend (proxied by nginx)
    const ML_SERVICE_URL = '/ml';  // Use relative path for ML service (proxied by nginx)

    // Initialize game
    function initGame() {
        // Set up canvas
        canvas.width = GAME_CONFIG.CANVAS_WIDTH;
        canvas.height = GAME_CONFIG.CANVAS_HEIGHT;
        
        // Event listeners
        document.addEventListener('keydown', handleKeyDown);
        document.addEventListener('keyup', handleKeyUp);
        
        // Camera controls
        const startCameraBtn = document.getElementById('startCamera');
        const stopCameraBtn = document.getElementById('stopCamera');
        const playAgainBtn = document.getElementById('playAgain');
        
        if (startCameraBtn) startCameraBtn.addEventListener('click', startCamera);
        if (stopCameraBtn) stopCameraBtn.addEventListener('click', stopCamera);
        if (playAgainBtn) playAgainBtn.addEventListener('click', resetGame);
        
        // Check model status
        checkModelStatus();
        
        // Start camera by default
        startCamera();
        
        // Start game loop
        gameLoop = requestAnimationFrame(updateGame);
        
        console.log('Flappy Bird Emotion Game initialized successfully');
    }

    // Model status check
    async function checkModelStatus() {
        try {
            const response = await fetch(`${ML_SERVICE_URL}/health`);
            if (response.ok) {
                const data = await response.json();
                if (data.model_loaded) {
                    modelStatusDisplay.textContent = '✅ Loaded';
                    modelStatusDisplay.className = 'loaded';
                } else {
                    modelStatusDisplay.textContent = '❌ Not Loaded';
                    modelStatusDisplay.className = 'not-loaded';
                }
            } else {
                modelStatusDisplay.textContent = '❌ Error';
                modelStatusDisplay.className = 'error';
            }
        } catch (error) {
            modelStatusDisplay.textContent = '❌ Offline';
            modelStatusDisplay.className = 'offline';
            console.error('Model status check error:', error);
        }
    }

    // Camera functions
    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: 320, 
                    height: 240,
                    facingMode: 'user'
                } 
            });
            video.srcObject = stream;
            startEmotionDetection();
            console.log('Camera started successfully');
        } catch (error) {
            console.log('Camera not available - game will run without emotion detection');
            emotionDisplay.textContent = 'Camera not available';
        }
    }

    function stopCamera() {
        const stream = video.srcObject;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        stopEmotionDetection();
    }

    function startEmotionDetection() {
        if (video.srcObject) {
            emotionDetectionInterval = setInterval(captureAndAnalyzeEmotion, gameState.emotionCheckInterval);
            console.log('Emotion detection started');
        } else {
            console.log('Emotion detection not started - camera not available');
        }
    }

    function stopEmotionDetection() {
        if (emotionDetectionInterval) {
            clearInterval(emotionDetectionInterval);
            emotionDetectionInterval = null;
        }
    }

    async function captureAndAnalyzeEmotion() {
        if (!video.srcObject) return;
        
        try {
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, 320, 240);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            console.log('📸 CAPTURING IMAGE:', new Date().toISOString());
            console.log('📤 Sending emotion request to:', `${ML_SERVICE_URL}/predict_emotion`);
            console.log('🖼️ Image size:', imageData.length, 'characters');
            console.log('📊 Canvas dimensions:', canvas.width, 'x', canvas.height);
            
            const response = await fetch(`${ML_SERVICE_URL}/predict_emotion`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image_data: imageData })
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('🤖 ML Service Response:', result);
                if (result.success) {
                    console.log(`🎭 Processing emotion: ${result.emotion} with confidence: ${result.confidence}`);
                    updateEmotion(result.emotion, result.confidence);
                } else {
                    console.log('❌ ML Service returned success: false');
                }
            } else {
                console.log('❌ ML Service response not ok:', response.status);
            }
        } catch (error) {
            console.log('Emotion detection not available - game will run with normal difficulty');
            console.error('Emotion detection error:', error);
            
            // Show error in emotion display
            emotionDisplay.textContent = 'Detection Error';
            emotionDisplay.className = 'error';
        }
    }

    function updateEmotion(emotion, confidence) {
        console.log(`updateEmotion called with: ${emotion}, confidence: ${confidence}`);
        
        // Accept emotions with lower confidence for better responsiveness
        if (confidence > 0.1) {
            gameState.currentEmotion = emotion;
            gameState.emotionsDetected++;
            
            // Update emotion display with confidence
            emotionDisplay.textContent = `${emotion} (${(confidence * 100).toFixed(1)}%)`;
            emotionDisplay.className = emotion;
            
            // Update difficulty based on emotion
            updateDifficulty();
            
            // Show emotion notification on canvas
            showEmotionNotification(emotion, confidence);
            
            console.log(`Emotion detected: ${emotion} (confidence: ${confidence.toFixed(2)})`);
        } else {
            // Show low confidence emotion but don't update game state
            emotionDisplay.textContent = `${emotion} (${(confidence * 100).toFixed(1)}%) - Low`;
            emotionDisplay.className = 'low-confidence';
            console.log(`Low confidence emotion: ${emotion} (confidence: ${confidence.toFixed(2)})`);
            
            // Still update difficulty for low confidence emotions
            updateDifficulty();
        }
    }

    // Game control functions
    function handleKeyDown(event) {
        if (event.code === 'Space' && !gameState.isRunning) {
            event.preventDefault();
            startGame();
        } else if (event.code === 'Space' && gameState.isRunning && !gameState.isGameOver) {
            event.preventDefault();
            flap();
        } else if (event.code === 'Escape' && gameState.isRunning) {
            event.preventDefault();
            togglePause();
        } else if (event.code === 'KeyR' && gameState.isGameOver) {
            event.preventDefault();
            resetGame();
        }
    }

    function handleKeyUp(event) {
        // No key up actions needed for Flappy Bird
    }

    function startGame() {
        gameState.isRunning = true;
        gameState.isPaused = false;
        gameState.isGameOver = false;
        gameState.score = 0;
        gameState.gameStartTime = Date.now();
        gameState.lastEmotionCheck = 0;
        gameState.currentEmotion = null;
        gameState.difficultyLevel = 1.0;
        gameState.maxDifficulty = 1.0;
        gameState.emotionsDetected = 0;
        
        // Reset difficulty parameters
        gameState.currentGravity = GAME_CONFIG.GRAVITY;
        gameState.currentFlapPower = GAME_CONFIG.FLAP_POWER;
        gameState.currentPipeGap = GAME_CONFIG.PIPE_GAP;
        gameState.currentPipeSpeed = GAME_CONFIG.PIPE_SPEED;
        gameState.currentPipeSpacing = GAME_CONFIG.PIPE_SPACING;
        
        // Reset bird
        bird.y = GAME_CONFIG.CANVAS_HEIGHT / 2;
        bird.velocity = 0;
        
        // Clear pipes
        pipes = [];
        
        // Start game session
        startGameSession();
        
        console.log('Flappy Bird game started');
    }

    function startGameSession() {
        fetch(`${BACKEND_URL}/game/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: `session_${Date.now()}`
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                gameState.sessionId = data.session_id;
                console.log('Game session started:', data.session_id);
            }
        })
        .catch(error => {
            console.error('Error starting game session:', error);
        });
    }

    function flap() {
        if (!gameState.isGameOver) {
            bird.velocity = gameState.currentFlapPower;
        }
    }

    function togglePause() {
        gameState.isPaused = !gameState.isPaused;
        if (gameState.isPaused) {
            cancelAnimationFrame(gameLoop);
        } else {
            gameLoop = requestAnimationFrame(updateGame);
        }
    }

    function updateDifficulty() {
        // Calculate base difficulty from score
        let baseDifficulty = 1.0 + (gameState.score / 100) * 0.2;
        
        // Apply emotion-based adjustments
        let emotionMultiplier = 1.0;
        let difficultyReason = "Normal";
        
        if (gameState.currentEmotion && gameState.currentEmotion !== 'Camera not available') {
            const emotionMultipliers = {
                'happy': 1.1,      // Slightly harder - engaged player
                'sad': 0.8,        // Easier - supportive gameplay
                'angry': 0.7,      // Much easier - reduce frustration
                'fear': 0.9,       // Easier - reduce stress
                'surprise': 1.0,   // Neutral
                'disgust': 1.0,    // Neutral
                'neutral': 1.0     // Neutral
            };
            emotionMultiplier = emotionMultipliers[gameState.currentEmotion] || 1.0;
            
            // Set difficulty reason
            const reasons = {
                'happy': "You're engaged!",
                'sad': "Supportive gameplay",
                'angry': "Reducing frustration",
                'fear': "Gentle encouragement",
                'surprise': "Neutral difficulty",
                'disgust': "Neutral difficulty",
                'neutral': "Balanced gameplay"
            };
            difficultyReason = reasons[gameState.currentEmotion] || "Normal";
        }
        
        gameState.difficultyLevel = Math.min(baseDifficulty * emotionMultiplier, 2.0);
        gameState.maxDifficulty = Math.max(gameState.maxDifficulty, gameState.difficultyLevel);
        
        // Update Flappy Bird specific parameters based on emotion
        updateFlappyBirdParameters();
        
        // Update display with reason
        difficultyDisplay.textContent = `${gameState.difficultyLevel.toFixed(1)} (${difficultyReason})`;
        
        // Show difficulty change notification
        showDifficultyNotification(gameState.difficultyLevel, difficultyReason);
    }

    function updateFlappyBirdParameters() {
        const emotion = gameState.currentEmotion;
        
        if (emotion === 'angry' || emotion === 'fear') {
            // Make game easier for frustrated or stressed players
            gameState.currentGravity = GAME_CONFIG.GRAVITY * 0.85;  // Less extreme
            gameState.currentFlapPower = GAME_CONFIG.FLAP_POWER * 1.1;  // Less extreme
            gameState.currentPipeGap = GAME_CONFIG.PIPE_GAP * 1.2;  // Less extreme
            gameState.currentPipeSpeed = GAME_CONFIG.PIPE_SPEED * 0.85;  // Less extreme
            gameState.currentPipeSpacing = GAME_CONFIG.PIPE_SPACING * 1.1;  // Less extreme
        } else if (emotion === 'sad') {
            // Make game easier for sad players
            gameState.currentGravity = GAME_CONFIG.GRAVITY * 0.9;
            gameState.currentFlapPower = GAME_CONFIG.FLAP_POWER * 1.05;  // Less extreme
            gameState.currentPipeGap = GAME_CONFIG.PIPE_GAP * 1.1;  // Less extreme
            gameState.currentPipeSpeed = GAME_CONFIG.PIPE_SPEED * 0.9;
            gameState.currentPipeSpacing = GAME_CONFIG.PIPE_SPACING * 1.05;  // Less extreme
        } else if (emotion === 'happy') {
            // Make game slightly harder for engaged players
            gameState.currentGravity = GAME_CONFIG.GRAVITY * 1.05;  // Less extreme
            gameState.currentFlapPower = GAME_CONFIG.FLAP_POWER * 0.95;  // Less extreme
            gameState.currentPipeGap = GAME_CONFIG.PIPE_GAP * 0.95;  // Less extreme
            gameState.currentPipeSpeed = GAME_CONFIG.PIPE_SPEED * 1.05;  // Less extreme
            gameState.currentPipeSpacing = GAME_CONFIG.PIPE_SPACING * 0.95;  // Less extreme
        } else {
            // Default parameters
            gameState.currentGravity = GAME_CONFIG.GRAVITY;
            gameState.currentFlapPower = GAME_CONFIG.FLAP_POWER;
            gameState.currentPipeGap = GAME_CONFIG.PIPE_GAP;
            gameState.currentPipeSpeed = GAME_CONFIG.PIPE_SPEED;
            gameState.currentPipeSpacing = GAME_CONFIG.PIPE_SPACING;
        }
    }

    // Game update functions
    function updateGame(currentTime) {
        if (!gameState.isRunning || gameState.isPaused) {
            gameLoop = requestAnimationFrame(updateGame);
            return;
        }

        const deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        // Update bird
        updateBird(deltaTime);

        // Update pipes
        updatePipes(deltaTime);

        // Check collisions
        checkCollisions();

        // Update score
        updateScore();

        // Update difficulty
        updateDifficulty();

        // Send game state to backend
        sendGameState();

        // Render
        render();

        // Continue game loop
        gameLoop = requestAnimationFrame(updateGame);
    }

    function updateBird(deltaTime) {
        // Apply gravity
        bird.velocity += gameState.currentGravity;
        bird.y += bird.velocity;

        // Keep bird within canvas bounds
        if (bird.y < 0) {
            bird.y = 0;
            bird.velocity = 0;
        }
        if (bird.y > GAME_CONFIG.CANVAS_HEIGHT - bird.size) {
            bird.y = GAME_CONFIG.CANVAS_HEIGHT - bird.size;
            bird.velocity = 0;
        }
    }

    function updatePipes(deltaTime) {
        // Move pipes
        for (let i = pipes.length - 1; i >= 0; i--) {
            pipes[i].x -= gameState.currentPipeSpeed;
            
            // Remove off-screen pipes
            if (pipes[i].x + GAME_CONFIG.PIPE_WIDTH < 0) {
                pipes.splice(i, 1);
            }
        }
        
        // Generate new pipes
        if (pipes.length === 0 || 
            pipes[pipes.length - 1].x < GAME_CONFIG.CANVAS_WIDTH - gameState.currentPipeSpacing) {
            generatePipe();
        }
    }

    function generatePipe() {
        const gapY = Math.random() * (GAME_CONFIG.CANVAS_HEIGHT - gameState.currentPipeGap - 100) + 50;
        
        pipes.push({
            x: GAME_CONFIG.CANVAS_WIDTH,
            gapY: gapY,
            gapHeight: gameState.currentPipeGap,
            passed: false
        });
    }

    function checkCollisions() {
        // Check if bird hits the ground or ceiling
        if (bird.y <= 0 || bird.y >= GAME_CONFIG.CANVAS_HEIGHT - bird.size) {
            gameOver();
            return;
        }

        // Check collision with pipes
        pipes.forEach(pipe => {
            // Check if bird passed the pipe
            if (!pipe.passed && bird.x > pipe.x + GAME_CONFIG.PIPE_WIDTH) {
                pipe.passed = true;
                gameState.score++;
                scoreDisplay.textContent = gameState.score;
            }

            // Check collision with pipe
            if (bird.x < pipe.x + GAME_CONFIG.PIPE_WIDTH &&
                bird.x + bird.size > pipe.x) {
                
                // Check top pipe
                if (bird.y < pipe.gapY) {
                    gameOver();
                    return;
                }
                
                // Check bottom pipe
                if (bird.y + bird.size > pipe.gapY + pipe.gapHeight) {
                    gameOver();
                    return;
                }
            }
        });
    }

    function updateScore() {
        // Score is updated in checkCollisions when passing pipes
    }

    function sendGameState() {
        if (!gameState.sessionId) return;
        
        fetch(`${BACKEND_URL}/game/update`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: gameState.sessionId,
                score: gameState.score,
                emotion_data: gameState.currentEmotion
            })
        })
        .catch(error => {
            console.error('Error sending game state:', error);
        });
    }

    // Rendering functions
    function render() {
        // Clear canvas
        ctx.clearRect(0, 0, GAME_CONFIG.CANVAS_WIDTH, GAME_CONFIG.CANVAS_HEIGHT);
        
        // Draw background
        drawBackground();
        
        // Draw pipes
        drawPipes();
        
        // Draw bird
        drawBird();
        
        // Draw UI
        drawUI();
    }

    function drawBackground() {
        // Sky gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, GAME_CONFIG.CANVAS_HEIGHT);
        gradient.addColorStop(0, '#87CEEB');
        gradient.addColorStop(1, '#98FB98');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, GAME_CONFIG.CANVAS_WIDTH, GAME_CONFIG.CANVAS_HEIGHT);
    }

    function drawPipes() {
        ctx.fillStyle = '#2ECC71';
        pipes.forEach(pipe => {
            // Top pipe
            ctx.fillRect(pipe.x, 0, GAME_CONFIG.PIPE_WIDTH, pipe.gapY);
            
            // Bottom pipe
            ctx.fillRect(pipe.x, pipe.gapY + pipe.gapHeight, GAME_CONFIG.PIPE_WIDTH, GAME_CONFIG.CANVAS_HEIGHT - pipe.gapY - pipe.gapHeight);
            
            // Pipe borders
            ctx.strokeStyle = '#27AE60';
            ctx.lineWidth = 3;
            ctx.strokeRect(pipe.x, 0, GAME_CONFIG.PIPE_WIDTH, pipe.gapY);
            ctx.strokeRect(pipe.x, pipe.gapY + pipe.gapHeight, GAME_CONFIG.PIPE_WIDTH, GAME_CONFIG.CANVAS_HEIGHT - pipe.gapY - pipe.gapHeight);
        });
    }

    function drawBird() {
        // Bird body
        ctx.fillStyle = '#F39C12';
        ctx.fillRect(bird.x, bird.y, bird.size, bird.size);
        
        // Bird eye
        ctx.fillStyle = 'white';
        ctx.fillRect(bird.x + 5, bird.y + 5, 8, 8);
        ctx.fillStyle = 'black';
        ctx.fillRect(bird.x + 7, bird.y + 7, 4, 4);
        
        // Bird wing
        ctx.fillStyle = '#E67E22';
        ctx.fillRect(bird.x + 15, bird.y + 10, 10, 8);
        
        // Bird border
        ctx.strokeStyle = '#D68910';
        ctx.lineWidth = 2;
        ctx.strokeRect(bird.x, bird.y, bird.size, bird.size);
    }

    function drawUI() {
        // Draw emotion indicator
        if (gameState.currentEmotion) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(10, 10, 200, 30);
            ctx.fillStyle = 'white';
            ctx.font = '16px Arial';
            ctx.fillText(`Emotion: ${gameState.currentEmotion}`, 15, 30);
        }

        // Draw score
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(GAME_CONFIG.CANVAS_WIDTH - 150, 10, 140, 30);
        ctx.fillStyle = 'white';
        ctx.font = '20px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`Score: ${gameState.score}`, GAME_CONFIG.CANVAS_WIDTH - 80, 30);
        ctx.textAlign = 'left';
        
        // Draw notifications
        if (emotionNotification.show) {
            const timeElapsed = Date.now() - emotionNotification.time;
            const alpha = Math.max(0, 1 - (timeElapsed / 3000));
            
            ctx.fillStyle = `rgba(52, 152, 219, ${alpha * 0.9})`;
            ctx.fillRect(10, 50, 300, 40);
            ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
            ctx.font = 'bold 18px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(emotionNotification.text, 160, 75);
            ctx.textAlign = 'left';
        }
        
        if (difficultyNotification.show) {
            const timeElapsed = Date.now() - difficultyNotification.time;
            const alpha = Math.max(0, 1 - (timeElapsed / 3000));
            
            ctx.fillStyle = `rgba(46, 204, 113, ${alpha * 0.9})`;
            ctx.fillRect(10, 100, 350, 40);
            ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
            ctx.font = 'bold 16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(difficultyNotification.text, 185, 125);
            ctx.textAlign = 'left';
        }
    }

    // Notification functions
    let emotionNotification = { show: false, text: '', time: 0 };
    let difficultyNotification = { show: false, text: '', time: 0 };
    
    function showEmotionNotification(emotion, confidence) {
        emotionNotification = {
            show: true,
            text: `😊 ${emotion.toUpperCase()} (${(confidence * 100).toFixed(1)}%)`,
            time: Date.now()
        };
        
        // Hide after 3 seconds
        setTimeout(() => {
            emotionNotification.show = false;
        }, 3000);
    }
    
    function showDifficultyNotification(difficulty, reason) {
        difficultyNotification = {
            show: true,
            text: `⚡ Difficulty: ${difficulty.toFixed(1)} - ${reason}`,
            time: Date.now()
        };
        
        // Hide after 3 seconds
        setTimeout(() => {
            difficultyNotification.show = false;
        }, 3000);
    }
    
    // Game over functions
    function gameOver() {
        gameState.isRunning = false;
        gameState.isGameOver = true;
        cancelAnimationFrame(gameLoop);
        
        // Send final score to backend
        endGameSession();
        
        // Show game over message
        showGameOver();
        
        // Show stats after a delay
        setTimeout(showGameStats, 2000);
    }

    function showGameOver() {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(0, 0, GAME_CONFIG.CANVAS_WIDTH, GAME_CONFIG.CANVAS_HEIGHT);
        
        ctx.fillStyle = 'white';
        ctx.font = '48px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('GAME OVER', GAME_CONFIG.CANVAS_WIDTH / 2, GAME_CONFIG.CANVAS_HEIGHT / 2 - 50);
        ctx.font = '24px Arial';
        ctx.fillText(`Score: ${gameState.score}`, GAME_CONFIG.CANVAS_WIDTH / 2, GAME_CONFIG.CANVAS_HEIGHT / 2);
        ctx.fillText('Press R to restart', GAME_CONFIG.CANVAS_WIDTH / 2, GAME_CONFIG.CANVAS_HEIGHT / 2 + 40);
        ctx.textAlign = 'left';
    }

    function showGameStats() {
        const gameDuration = Math.floor((Date.now() - gameState.gameStartTime) / 1000);
        
        document.getElementById('finalScore').textContent = gameState.score;
        document.getElementById('maxDifficulty').textContent = gameState.maxDifficulty.toFixed(1);
        document.getElementById('emotionsDetected').textContent = gameState.emotionsDetected;
        document.getElementById('gameDuration').textContent = `${gameDuration}s`;
        
        statsSection.style.display = 'block';
    }

    function endGameSession() {
        if (!gameState.sessionId) return;
        
        fetch(`${BACKEND_URL}/game/end`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: gameState.sessionId,
                final_score: gameState.score
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('Game session ended:', data.session_stats);
            }
        })
        .catch(error => {
            console.error('Error ending game session:', error);
        });
    }

    function resetGame() {
        statsSection.style.display = 'none';
        emotionDisplay.textContent = 'None';
        emotionDisplay.className = '';
        scoreDisplay.textContent = '0';
        difficultyDisplay.textContent = '1.0';
        
        // Reset game state
        gameState = {
            isRunning: false,
            isPaused: false,
            isGameOver: false,
            score: 0,
            sessionId: null,
            gameStartTime: null,
            lastEmotionCheck: 0,
            emotionCheckInterval: 3500,
            currentEmotion: null,
            difficultyLevel: 1.0,
            maxDifficulty: 1.0,
            emotionsDetected: 0,
            currentGravity: GAME_CONFIG.GRAVITY,
            currentFlapPower: GAME_CONFIG.FLAP_POWER,
            currentPipeGap: GAME_CONFIG.PIPE_GAP,
            currentPipeSpeed: GAME_CONFIG.PIPE_SPEED,
            currentPipeSpacing: GAME_CONFIG.PIPE_SPACING
        };
        
        // Reset bird
        bird.y = GAME_CONFIG.CANVAS_HEIGHT / 2;
        bird.velocity = 0;
        
        // Clear pipes
        pipes = [];
        
        // Restart game loop
        gameLoop = requestAnimationFrame(updateGame);
    }

    // Initialize game when page loads
    window.addEventListener('load', initGame);
})(); 