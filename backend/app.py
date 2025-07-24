#!/usr/bin/env python3
"""
Flappy Bird Emotion Game - Backend API
Flask application for game session management and statistics
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Game sessions storage (in production, use a database)
game_sessions = {}
game_statistics = {
    'total_games': 0,
    'total_score': 0,
    'average_score': 0,
    'highest_score': 0,
    'total_emotions_detected': 0,
    'emotion_frequency': {
        'happy': 0,
        'sad': 0,
        'angry': 0,
        'fear': 0,
        'surprise': 0,
        'disgust': 0,
        'neutral': 0
    }
}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'flappy-bird-backend',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/game/start', methods=['POST'])
def start_game():
    """Start a new game session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID required'}), 400
        
        # Initialize session
        game_sessions[session_id] = {
            'session_id': session_id,
            'start_time': time.time(),
            'score': 0,
            'emotions_detected': 0,
            'emotion_history': [],
            'difficulty_progression': [],
            'is_active': True
        }
        
        logger.info(f"Game session started: {session_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Game session started successfully'
        })
        
    except Exception as e:
        logger.error(f"Error starting game session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/game/update', methods=['POST'])
def update_game():
    """Update game state and get difficulty adjustment"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        score = data.get('score', 0)
        emotion_data = data.get('emotion_data')
        
        if not session_id or session_id not in game_sessions:
            return jsonify({'success': False, 'error': 'Invalid session ID'}), 400
        
        session = game_sessions[session_id]
        
        # Update session data
        session['score'] = score
        session['last_update'] = time.time()
        
        # Record emotion if provided
        if emotion_data:
            session['emotions_detected'] += 1
            session['emotion_history'].append({
                'emotion': emotion_data,
                'timestamp': time.time(),
                'score': score
            })
            
            # Update global statistics
            if emotion_data in game_statistics['emotion_frequency']:
                game_statistics['emotion_frequency'][emotion_data] += 1
            game_statistics['total_emotions_detected'] += 1
        
        # Calculate difficulty adjustment based on emotion
        difficulty_multiplier = calculate_difficulty_adjustment(emotion_data, score)
        
        # Record difficulty progression
        session['difficulty_progression'].append({
            'score': score,
            'difficulty': difficulty_multiplier,
            'emotion': emotion_data,
            'timestamp': time.time()
        })
        
        return jsonify({
            'success': True,
            'difficulty_multiplier': difficulty_multiplier,
            'session_stats': {
                'score': score,
                'emotions_detected': session['emotions_detected'],
                'session_duration': time.time() - session['start_time']
            }
        })
        
    except Exception as e:
        logger.error(f"Error updating game: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/game/end', methods=['POST'])
def end_game():
    """End game session and return statistics"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        final_score = data.get('final_score', 0)
        
        if not session_id or session_id not in game_sessions:
            return jsonify({'success': False, 'error': 'Invalid session ID'}), 400
        
        session = game_sessions[session_id]
        
        # Update session with final data
        session['end_time'] = time.time()
        session['final_score'] = final_score
        session['is_active'] = False
        session['duration'] = session['end_time'] - session['start_time']
        
        # Update global statistics
        game_statistics['total_games'] += 1
        game_statistics['total_score'] += final_score
        game_statistics['average_score'] = game_statistics['total_score'] / game_statistics['total_games']
        
        if final_score > game_statistics['highest_score']:
            game_statistics['highest_score'] = final_score
        
        # Calculate session statistics
        session_stats = calculate_session_statistics(session)
        
        logger.info(f"Game session ended: {session_id}, Score: {final_score}")
        
        return jsonify({
            'success': True,
            'session_stats': session_stats,
            'global_stats': game_statistics
        })
        
    except Exception as e:
        logger.error(f"Error ending game: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/game/stats', methods=['GET'])
def get_game_statistics():
    """Get overall game statistics"""
    try:
        # Calculate additional statistics
        active_sessions = len([s for s in game_sessions.values() if s.get('is_active', False)])
        
        stats = {
            'global_statistics': game_statistics,
            'active_sessions': active_sessions,
            'total_sessions': len(game_sessions),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def calculate_difficulty_adjustment(emotion, score):
    """
    Calculate difficulty adjustment based on emotion and score
    Returns a multiplier for game difficulty
    """
    # Base difficulty increases with score
    base_difficulty = 1.0 + (score / 100) * 0.2
    
    # Emotion-based adjustments
    emotion_multipliers = {
        'happy': 1.1,      # Slightly harder - engaged player
        'sad': 0.8,        # Easier - supportive gameplay
        'angry': 0.7,      # Much easier - reduce frustration
        'fear': 0.9,       # Easier - reduce stress
        'surprise': 1.0,   # Neutral
        'disgust': 1.0,    # Neutral
        'neutral': 1.0     # Neutral
    }
    
    emotion_multiplier = emotion_multipliers.get(emotion, 1.0)
    
    # Calculate final difficulty
    final_difficulty = base_difficulty * emotion_multiplier
    
    # Clamp between 0.5 and 2.0
    return max(0.5, min(2.0, final_difficulty))

def calculate_session_statistics(session):
    """Calculate detailed statistics for a game session"""
    duration = session.get('duration', 0)
    emotions = session.get('emotion_history', [])
    
    # Emotion frequency in this session
    session_emotions = {}
    for emotion_record in emotions:
        emotion = emotion_record['emotion']
        session_emotions[emotion] = session_emotions.get(emotion, 0) + 1
    
    # Calculate average difficulty
    difficulties = [d['difficulty'] for d in session.get('difficulty_progression', [])]
    avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 1.0
    
    return {
        'session_id': session['session_id'],
        'final_score': session.get('final_score', 0),
        'duration_seconds': round(duration, 2),
        'emotions_detected': session.get('emotions_detected', 0),
        'emotion_frequency': session_emotions,
        'average_difficulty': round(avg_difficulty, 2),
        'max_difficulty': max(difficulties) if difficulties else 1.0,
        'min_difficulty': min(difficulties) if difficulties else 1.0,
        'start_time': datetime.fromtimestamp(session['start_time']).isoformat(),
        'end_time': datetime.fromtimestamp(session['end_time']).isoformat() if 'end_time' in session else None
    }

@app.route('/api/game/sessions/<session_id>', methods=['GET'])
def get_session_details(session_id):
    """Get detailed information about a specific session"""
    try:
        if session_id not in game_sessions:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        session = game_sessions[session_id]
        session_stats = calculate_session_statistics(session)
        
        return jsonify({
            'success': True,
            'session': session_stats
        })
        
    except Exception as e:
        logger.error(f"Error getting session details: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/game/emotions', methods=['GET'])
def get_emotion_statistics():
    """Get emotion detection statistics"""
    try:
        emotion_stats = {
            'total_emotions_detected': game_statistics['total_emotions_detected'],
            'emotion_frequency': game_statistics['emotion_frequency'],
            'emotion_percentages': {}
        }
        
        # Calculate percentages
        total = sum(game_statistics['emotion_frequency'].values())
        if total > 0:
            for emotion, count in game_statistics['emotion_frequency'].items():
                emotion_stats['emotion_percentages'][emotion] = round((count / total) * 100, 2)
        
        return jsonify({
            'success': True,
            'emotion_statistics': emotion_stats
        })
        
    except Exception as e:
        logger.error(f"Error getting emotion statistics: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5002))
    
    logger.info(f"Starting Flappy Bird Emotion Game Backend on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True) 