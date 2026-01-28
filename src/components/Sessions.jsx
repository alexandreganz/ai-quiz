import React, { useState, useEffect } from 'react';
import { getAllSessions, deleteSession } from '../services/storage';

function Sessions({ onBackToHome, onViewSession }) {
  const [sessions, setSessions] = useState([]);
  const [filterScore, setFilterScore] = useState('all'); // all, excellent (80+), good (60-79), needs-improvement (<60)
  const [sortBy, setSortBy] = useState('date'); // date, score

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = () => {
    const allSessions = getAllSessions();
    setSessions(allSessions);
  };

  const handleDelete = (sessionId, event) => {
    event.stopPropagation();
    if (window.confirm('Are you sure you want to delete this session?')) {
      deleteSession(sessionId);
      loadSessions();
    }
  };

  const formatDate = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getScoreColor = (score, total) => {
    const percentage = (score / total) * 100;
    if (percentage >= 80) return 'text-green-600';
    if (percentage >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBgColor = (score, total) => {
    const percentage = (score / total) * 100;
    if (percentage >= 80) return 'bg-green-50 border-green-200';
    if (percentage >= 60) return 'bg-yellow-50 border-yellow-200';
    return 'bg-red-50 border-red-200';
  };

  const getFilteredAndSortedSessions = () => {
    let filtered = [...sessions];

    // Apply score filter
    if (filterScore !== 'all') {
      filtered = filtered.filter(session => {
        const percentage = (session.score / session.totalQuestions) * 100;
        if (filterScore === 'excellent') return percentage >= 80;
        if (filterScore === 'good') return percentage >= 60 && percentage < 80;
        if (filterScore === 'needs-improvement') return percentage < 60;
        return true;
      });
    }

    // Apply sorting
    if (sortBy === 'score') {
      filtered.sort((a, b) => {
        const percentageA = (a.score / a.totalQuestions) * 100;
        const percentageB = (b.score / b.totalQuestions) * 100;
        return percentageB - percentageA; // Highest first
      });
    } else {
      // Sort by date (newest first)
      filtered.sort((a, b) => new Date(b.date) - new Date(a.date));
    }

    return filtered;
  };

  const filteredSessions = getFilteredAndSortedSessions();

  if (sessions.length === 0) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-2xl p-12 text-center max-w-md">
          <div className="text-6xl mb-4">📊</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">No Sessions Yet</h2>
          <p className="text-gray-600 mb-6">
            Complete a quiz and save your session to see it here.
          </p>
          <button
            onClick={onBackToHome}
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-xl transition duration-200"
          >
            Back to Home
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4 py-8">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Quiz Sessions</h1>
              <p className="text-gray-600 mt-1">
                {sessions.length} session{sessions.length !== 1 ? 's' : ''} completed
                {filteredSessions.length !== sessions.length && ` • Showing ${filteredSessions.length}`}
              </p>
            </div>
            <button
              onClick={onBackToHome}
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-6 rounded-xl transition duration-200"
            >
              Back to Home
            </button>
          </div>

          {/* Filters */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-gray-200">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Filter by Score
              </label>
              <select
                value={filterScore}
                onChange={(e) => setFilterScore(e.target.value)}
                className="w-full px-4 py-2 border-2 border-gray-200 rounded-lg focus:border-indigo-500 focus:outline-none"
              >
                <option value="all">All Sessions</option>
                <option value="excellent">Excellent (80%+)</option>
                <option value="good">Good (60-79%)</option>
                <option value="needs-improvement">Needs Improvement (&lt;60%)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Sort by
              </label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="w-full px-4 py-2 border-2 border-gray-200 rounded-lg focus:border-indigo-500 focus:outline-none"
              >
                <option value="date">Date (Newest First)</option>
                <option value="score">Score (Highest First)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Sessions List */}
        {filteredSessions.length === 0 ? (
          <div className="bg-white rounded-xl shadow-lg p-12 text-center">
            <div className="text-6xl mb-4">🔍</div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">No sessions match your filters</h2>
            <p className="text-gray-600">Try adjusting your filter settings</p>
          </div>
        ) : (
          <div className="space-y-4">
            {filteredSessions.map((session, index) => {
            const percentage = Math.round((session.score / session.totalQuestions) * 100);

            return (
              <div
                key={session.id}
                onClick={() => onViewSession(session.id)}
                className={`bg-white rounded-xl shadow-lg p-6 cursor-pointer transition-all duration-200 hover:shadow-xl hover:-translate-y-1 border-2 ${getScoreBgColor(session.score, session.totalQuestions)}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-4 mb-2">
                      <h3 className="text-xl font-bold text-gray-900">
                        Session #{sessions.length - index}
                      </h3>
                      <span className="text-sm text-gray-500">
                        {formatDate(session.date)}
                      </span>
                    </div>

                    {/* Score Display */}
                    <div className="flex items-center gap-6">
                      <div>
                        <span className={`text-3xl font-bold ${getScoreColor(session.score, session.totalQuestions)}`}>
                          {session.score}/{session.totalQuestions}
                        </span>
                        <span className="text-gray-600 ml-2">correct</span>
                      </div>
                      <div>
                        <span className={`text-3xl font-bold ${getScoreColor(session.score, session.totalQuestions)}`}>
                          {percentage}%
                        </span>
                      </div>
                    </div>

                    {/* Category breakdown */}
                    <div className="mt-4 flex flex-wrap gap-2">
                      {['LLM', 'LLMOps', 'GenAI', 'Forecasting'].map(category => {
                        const categoryQuestions = session.questions.filter(q => q.category === category);
                        const categoryCorrect = categoryQuestions.filter((q, idx) => {
                          const questionIndex = session.questions.indexOf(q);
                          return session.answers[questionIndex] === q.correctAnswer;
                        }).length;

                        return categoryQuestions.length > 0 && (
                          <span
                            key={category}
                            className={`px-3 py-1 rounded-full text-sm font-semibold ${
                              category === 'LLM' ? 'bg-blue-100 text-blue-800' :
                              category === 'LLMOps' ? 'bg-purple-100 text-purple-800' :
                              category === 'GenAI' ? 'bg-indigo-100 text-indigo-800' :
                              'bg-orange-100 text-orange-800'
                            }`}
                          >
                            {category}: {categoryCorrect}/{categoryQuestions.length}
                          </span>
                        );
                      })}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-3 ml-4">
                    <button
                      onClick={(e) => handleDelete(session.id, e)}
                      className="bg-red-100 hover:bg-red-200 text-red-600 font-semibold py-2 px-4 rounded-lg transition duration-200"
                    >
                      Delete
                    </button>
                    <div className="text-indigo-600 text-2xl">→</div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        )}
      </div>
    </div>
  );
}

export default Sessions;
