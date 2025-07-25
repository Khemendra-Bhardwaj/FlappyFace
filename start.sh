#!/bin/bash

# Flappy Bird Emotion Game - Startup Script

echo "🐦 Starting Flappy Bird Emotion Game..."
echo "======================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down --remove-orphans

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo "📊 Checking service status..."
docker-compose ps

# Test services
echo "🧪 Testing services..."

# Test frontend
if curl -s http://localhost:3000/health > /dev/null; then
    echo "✅ Frontend is running on http://localhost:3000"
else
    echo "❌ Frontend is not responding"
fi

# Test backend
if curl -s http://localhost:5002/health > /dev/null; then
    echo "✅ Backend is running on http://localhost:5002"
else
    echo "❌ Backend is not responding"
fi

# Test ML service
if curl -s http://localhost:5003/health > /dev/null; then
    echo "✅ ML Service is running on http://localhost:5003"
else
    echo "❌ ML Service is not responding"
fi

# Test ML service via proxy
if curl -s http://localhost:3000/ml/test > /dev/null; then
    echo "✅ ML Service proxy is working"
else
    echo "❌ ML Service proxy is not working"
fi

echo ""
echo "🎮 Flappy Bird Emotion Game is ready!"
echo "🌐 Open your browser and go to: http://localhost:3000"
echo ""
echo "📋 Quick Start:"
echo "1. Allow camera access when prompted"
echo "2. Press SPACE to start the game"
echo "3. Press SPACE to flap the bird"
echo "4. Watch the difficulty change based on your emotions!"
echo ""
echo "🔧 To stop the game: docker-compose down"
echo "📊 To view logs: docker-compose logs -f" 