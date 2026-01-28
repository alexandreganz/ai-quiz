import React, { useState } from 'react';
import { getStatistics } from '../services/storage';

function Home({ onStartQuiz, onViewSessions }) {
  const stats = getStatistics();
  const [selectedCategories, setSelectedCategories] = useState(['LLM', 'LLMOps', 'GenAI', 'Forecasting']);

  const categories = [
    { id: 'LLM', name: 'LLM', description: 'Large Language Models' },
    { id: 'LLMOps', name: 'LLMOps', description: 'LLM Operations' },
    { id: 'GenAI', name: 'GenAI', description: 'Generative AI' },
    { id: 'Forecasting', name: 'Forecasting', description: 'Time Series & Demand Planning' }
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
    onStartQuiz(selectedCategories);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        <div className="bg-white rounded-2xl shadow-2xl p-8 md:p-12">
          <div className="text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-3">
              AI Expert Quiz
            </h1>
            <p className="text-lg text-gray-600">
              Test your knowledge on LLM, LLMOps, GenAI, and Forecasting
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

          {/* Quiz Info */}
          <div className="bg-gray-50 rounded-xl p-6 mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Quiz Format
            </h2>
            <ul className="space-y-2 text-gray-700">
              <li className="flex items-start">
                <span className="text-indigo-600 mr-2">•</span>
                <span>30 questions per quiz</span>
              </li>
              <li className="flex items-start">
                <span className="text-indigo-600 mr-2">•</span>
                <span>Mix of Easy, Medium, and Hard difficulty</span>
              </li>
              <li className="flex items-start">
                <span className="text-indigo-600 mr-2">•</span>
                <span>Topics: Choose from LLM, LLMOps, GenAI, and Forecasting</span>
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
              className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-4 px-6 rounded-xl transition duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
            >
              Start New Quiz
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
