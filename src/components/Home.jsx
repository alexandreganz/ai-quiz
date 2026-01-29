import React, { useState } from 'react';
import { getStatistics, getUsedQuestionIds, resetUsedQuestions } from '../services/storage';
import { questions } from '../data/questions';

function Home({ onStartQuiz, onViewSessions }) {
  const stats = getStatistics();
  const usedQuestionIds = getUsedQuestionIds();
  const [selectedCategories, setSelectedCategories] = useState(['LLM', 'LLMOps', 'GenAI', 'Forecasting', 'Databricks', 'Forecasting Enhancement']);
  const [easyCount, setEasyCount] = useState(10);
  const [mediumCount, setMediumCount] = useState(10);
  const [hardCount, setHardCount] = useState(10);

  const categories = [
    { id: 'LLM', name: 'LLM', description: 'Large Language Models' },
    { id: 'LLMOps', name: 'LLMOps', description: 'LLM Operations' },
    { id: 'GenAI', name: 'GenAI', description: 'Generative AI' },
    { id: 'Forecasting', name: 'Forecasting', description: 'Time Series & Demand Planning' },
    { id: 'Databricks', name: 'Databricks', description: 'Orchestration & PySpark Best Practices' },
    { id: 'Forecasting Enhancement', name: 'Forecasting Enhancement', description: 'Model Improvements & Adoption' }
  ];

  const toggleCategory = (categoryId) => {
    setSelectedCategories(prev => {
      if (prev.includes(categoryId)) {
        // Don't allow deselecting if it's the last one
        if (prev.length === 1) return prev;
        return prev.filter(id => id !== categoryId);
      } else {
        return [...prev, categoryId];
      }
    });
  };

  const handleStartQuiz = () => {
    if (totalAvailableCount < totalQuestions) {
      alert(`Not enough available questions! You need ${totalQuestions} questions but only ${totalAvailableCount} are available. Please reduce the number of questions or reset the question pool.`);
      return;
    }
    onStartQuiz(selectedCategories, { easy: easyCount, medium: mediumCount, hard: hardCount });
  };

  const handleResetQuestions = () => {
    if (window.confirm('Are you sure you want to reset? This will make all questions available again for future quizzes.')) {
      resetUsedQuestions();
      window.location.reload();
    }
  };

  const totalQuestions = easyCount + mediumCount + hardCount;

  // Calculate available questions
  const availableQuestions = questions.filter(q =>
    selectedCategories.includes(q.category) && !usedQuestionIds.includes(q.id)
  );
  const totalAvailableCount = availableQuestions.length;
  const usedCount = usedQuestionIds.length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        <div className="bg-white rounded-2xl shadow-2xl p-8 md:p-12">
          <div className="text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-3">
              AI Expert Quiz
            </h1>
            <p className="text-lg text-gray-600">
              Test your knowledge on LLM, LLMOps, GenAI, Forecasting, Databricks, and more
            </p>
          </div>

          {/* Statistics */}
          {stats.totalAttempts > 0 && (
            <div className="grid grid-cols-3 gap-4 mb-8 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl">
              <div className="text-center">
                <div className="text-3xl font-bold text-indigo-600">
                  {stats.totalAttempts}
                </div>
                <div className="text-sm text-gray-600 mt-1">Attempts</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-indigo-600">
                  {stats.averageScore}%
                </div>
                <div className="text-sm text-gray-600 mt-1">Avg Score</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-indigo-600">
                  {stats.bestScore}/30
                </div>
                <div className="text-sm text-gray-600 mt-1">Best Score</div>
              </div>
            </div>
          )}

          {/* Question Pool Status */}
          <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900">
                Question Pool Status
              </h2>
              {usedCount > 0 && (
                <button
                  onClick={handleResetQuestions}
                  className="bg-purple-600 hover:bg-purple-700 text-white text-sm font-semibold py-2 px-4 rounded-lg transition duration-200 shadow-md hover:shadow-lg"
                >
                  Reset All Questions
                </button>
              )}
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-white rounded-lg shadow-sm">
                <div className="text-3xl font-bold text-green-600">
                  {totalAvailableCount}
                </div>
                <div className="text-sm text-gray-600 mt-1">Available</div>
              </div>
              <div className="text-center p-4 bg-white rounded-lg shadow-sm">
                <div className="text-3xl font-bold text-orange-600">
                  {usedCount}
                </div>
                <div className="text-sm text-gray-600 mt-1">Used</div>
              </div>
            </div>
            {totalAvailableCount < totalQuestions && (
              <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                <p className="text-sm text-yellow-800">
                  ⚠️ Warning: Only {totalAvailableCount} questions available for selected categories. You requested {totalQuestions} questions.
                </p>
              </div>
            )}
            {usedCount > 0 && (
              <div className="mt-4 text-sm text-gray-600">
                <p>
                  You've used {usedCount} question{usedCount !== 1 ? 's' : ''} in previous sessions.
                  These won't appear in new quizzes unless you reset.
                </p>
              </div>
            )}
          </div>

          {/* Category Selection */}
          <div className="bg-gray-50 rounded-xl p-6 mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Select Categories
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {categories.map(category => (
                <label
                  key={category.id}
                  className={`flex items-center p-4 rounded-lg border-2 cursor-pointer transition-all ${
                    selectedCategories.includes(category.id)
                      ? 'border-indigo-500 bg-indigo-50'
                      : 'border-gray-200 bg-white hover:border-gray-300'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selectedCategories.includes(category.id)}
                    onChange={() => toggleCategory(category.id)}
                    className="w-5 h-5 text-indigo-600 rounded focus:ring-indigo-500 focus:ring-2"
                  />
                  <div className="ml-3">
                    <div className="font-semibold text-gray-900">{category.name}</div>
                    <div className="text-sm text-gray-600">{category.description}</div>
                  </div>
                </label>
              ))}
            </div>
            <p className="text-sm text-gray-500 mt-3">
              {selectedCategories.length} {selectedCategories.length === 1 ? 'category' : 'categories'} selected
            </p>
          </div>

          {/* Difficulty Distribution */}
          <div className="bg-gray-50 rounded-xl p-6 mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Difficulty Distribution
            </h2>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-semibold text-gray-700">Easy Questions</label>
                  <span className="text-sm font-bold text-green-600">{easyCount}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="30"
                  value={easyCount}
                  onChange={(e) => setEasyCount(parseInt(e.target.value))}
                  className="w-full h-2 bg-green-200 rounded-lg appearance-none cursor-pointer accent-green-600"
                />
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-semibold text-gray-700">Medium Questions</label>
                  <span className="text-sm font-bold text-yellow-600">{mediumCount}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="30"
                  value={mediumCount}
                  onChange={(e) => setMediumCount(parseInt(e.target.value))}
                  className="w-full h-2 bg-yellow-200 rounded-lg appearance-none cursor-pointer accent-yellow-600"
                />
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-semibold text-gray-700">Hard Questions</label>
                  <span className="text-sm font-bold text-red-600">{hardCount}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="30"
                  value={hardCount}
                  onChange={(e) => setHardCount(parseInt(e.target.value))}
                  className="w-full h-2 bg-red-200 rounded-lg appearance-none cursor-pointer accent-red-600"
                />
              </div>
              <div className="pt-3 border-t border-gray-300">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-gray-900">Total Questions:</span>
                  <span className="text-xl font-bold text-indigo-600">{totalQuestions}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Quiz Info */}
          <div className="bg-gray-50 rounded-xl p-6 mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Quiz Format
            </h2>
            <ul className="space-y-2 text-gray-700">
              <li className="flex items-start">
                <span className="text-indigo-600 mr-2">•</span>
                <span>{totalQuestions} questions per quiz</span>
              </li>
              <li className="flex items-start">
                <span className="text-indigo-600 mr-2">•</span>
                <span>Mix of Easy, Medium, and Hard difficulty</span>
              </li>
              <li className="flex items-start">
                <span className="text-indigo-600 mr-2">•</span>
                <span>Topics: Choose from LLM, LLMOps, GenAI, Forecasting, Databricks, and Forecasting Enhancement</span>
              </li>
              <li className="flex items-start">
                <span className="text-indigo-600 mr-2">•</span>
                <span>View detailed explanations after completion</span>
              </li>
              <li className="flex items-start">
                <span className="text-indigo-600 mr-2">•</span>
                <span>Track your progress over time</span>
              </li>
            </ul>
          </div>

          {/* Action Buttons */}
          <div className="space-y-4">
            <button
              onClick={handleStartQuiz}
              disabled={totalAvailableCount < totalQuestions}
              className={`w-full text-white font-semibold py-4 px-6 rounded-xl transition duration-200 shadow-lg transform ${
                totalAvailableCount < totalQuestions
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-xl hover:-translate-y-0.5'
              }`}
            >
              Start New Quiz {totalAvailableCount < totalQuestions && '(Not Enough Questions)'}
            </button>

            {stats.totalAttempts > 0 && (
              <button
                onClick={onViewSessions}
                className="w-full bg-white hover:bg-gray-50 text-indigo-600 font-semibold py-4 px-6 rounded-xl border-2 border-indigo-600 transition duration-200 transform hover:-translate-y-0.5"
              >
                View Past Sessions
              </button>
            )}
          </div>

          {/* Footer */}
          <div className="mt-8 text-center text-sm text-gray-500">
            Good luck with your AI Expert interview preparation!
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;
