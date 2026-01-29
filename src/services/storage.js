// localStorage service for managing quiz sessions

const SESSIONS_KEY = 'ai_quiz_sessions';
const USED_QUESTIONS_KEY = 'ai_quiz_used_questions';

// Generate unique ID for sessions
export function generateId() {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Get all sessions
export function getAllSessions() {
  try {
    const sessions = localStorage.getItem(SESSIONS_KEY);
    return sessions ? JSON.parse(sessions) : [];
  } catch (error) {
    console.error('Error reading sessions:', error);
    return [];
  }
}

// Save a new session
export function saveSession(sessionData) {
  try {
    const sessions = getAllSessions();
    const newSession = {
      id: generateId(),
      date: new Date().toISOString(),
      ...sessionData
    };
    sessions.unshift(newSession); // Add to beginning
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
    return newSession;
  } catch (error) {
    console.error('Error saving session:', error);
    throw error;
  }
}

// Get a specific session by ID
export function getSessionById(sessionId) {
  try {
    const sessions = getAllSessions();
    return sessions.find(session => session.id === sessionId);
  } catch (error) {
    console.error('Error getting session:', error);
    return null;
  }
}

// Delete a session
export function deleteSession(sessionId) {
  try {
    const sessions = getAllSessions();
    const filtered = sessions.filter(session => session.id !== sessionId);
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(filtered));
    return true;
  } catch (error) {
    console.error('Error deleting session:', error);
    return false;
  }
}

// Get statistics
export function getStatistics() {
  const sessions = getAllSessions();
  if (sessions.length === 0) {
    return {
      totalAttempts: 0,
      averageScore: 0,
      bestScore: 0,
      totalQuestions: 0
    };
  }

  const totalQuestions = sessions.reduce((sum, s) => sum + s.totalQuestions, 0);
  const totalCorrect = sessions.reduce((sum, s) => sum + s.score, 0);
  const scores = sessions.map(s => s.score);

  return {
    totalAttempts: sessions.length,
    averageScore: Math.round((totalCorrect / totalQuestions) * 100),
    bestScore: Math.max(...scores),
    totalQuestions: totalQuestions
  };
}

// Get used question IDs
export function getUsedQuestionIds() {
  try {
    const usedQuestions = localStorage.getItem(USED_QUESTIONS_KEY);
    return usedQuestions ? JSON.parse(usedQuestions) : [];
  } catch (error) {
    console.error('Error reading used questions:', error);
    return [];
  }
}

// Add question IDs to used list
export function addUsedQuestions(questionIds) {
  try {
    const usedQuestions = getUsedQuestionIds();
    const newUsedQuestions = [...new Set([...usedQuestions, ...questionIds])]; // Remove duplicates
    localStorage.setItem(USED_QUESTIONS_KEY, JSON.stringify(newUsedQuestions));
    return newUsedQuestions;
  } catch (error) {
    console.error('Error adding used questions:', error);
    throw error;
  }
}

// Reset used questions (make all questions available again)
export function resetUsedQuestions() {
  try {
    localStorage.setItem(USED_QUESTIONS_KEY, JSON.stringify([]));
    return true;
  } catch (error) {
    console.error('Error resetting used questions:', error);
    return false;
  }
}
