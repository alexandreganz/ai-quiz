import React from 'react';
import { saveSession, addUsedQuestions } from '../services/storage';

function Results({ questions, answers, onBackToHome, onViewSessions, sessionSaved = false }) {
  const score = answers.reduce((total, answer, index) => {
    return total + (answer === questions[index].correctAnswer ? 1 : 0);
  }, 0);

  const percentage = Math.round((score / questions.length) * 100);

  const handleSaveSession = () => {
    // Extract question IDs and mark them as used
    const questionIds = questions.map(q => q.id);
    addUsedQuestions(questionIds);

    saveSession({
      questions,
      answers,
      score,
      totalQuestions: questions.length,
      questionIds // Store question IDs in the session for reference
    });
    // Reload to show updated state with save button removed
    window.location.reload();
  };

  const getScoreColor = () => {
    if (percentage >= 80) return 'text-green-600';
    if (percentage >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreMessage = () => {
    if (percentage >= 90) return 'Outstanding! You\'re an AI expert!';
    if (percentage >= 80) return 'Excellent work! You know your stuff!';
    if (percentage >= 70) return 'Good job! Keep practicing!';
    if (percentage >= 60) return 'Not bad! Room for improvement!';
    return 'Keep studying! You\'ll get there!';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4 py-8">
      <div className="max-w-4xl mx-auto">
        {/* Score Card */}
        <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8">
          <div className="text-center mb-6">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Quiz Complete!</h1>
            <p className="text-lg text-gray-600">{getScoreMessage()}</p>
          </div>

          <div className="flex justify-center items-center gap-8 mb-6">
            <div className="text-center">
              <div className={`text-6xl font-bold ${getScoreColor()}`}>
                {score}/{questions.length}
              </div>
              <div className="text-gray-600 mt-2">Correct Answers</div>
            </div>
            <div className="text-center">
              <div className={`text-6xl font-bold ${getScoreColor()}`}>
                {percentage}%
              </div>
              <div className="text-gray-600 mt-2">Score</div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-4 justify-center">
            {!sessionSaved && (
              <button
                onClick={handleSaveSession}
                className="bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-6 rounded-xl transition duration-200 shadow-lg"
              >
                Save Session
              </button>
            )}
            <button
              onClick={onBackToHome}
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-xl transition duration-200 shadow-lg"
            >
              Back to Home
            </button>
            <button
              onClick={onViewSessions}
              className="bg-white hover:bg-gray-50 text-indigo-600 font-semibold py-3 px-6 rounded-xl border-2 border-indigo-600 transition duration-200"
            >
              View Sessions
            </button>
          </div>
        </div>

        {/* Detailed Results */}
        <div className="space-y-4">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Detailed Review</h2>

          {questions.map((question, index) => {
            const userAnswer = answers[index];
            const isCorrect = userAnswer === question.correctAnswer;

            return (
              <div
                key={question.id}
                className={`bg-white rounded-xl shadow-lg p-6 border-l-4 ${
                  isCorrect ? 'border-green-500' : 'border-red-500'
                }`}
              >
                {/* Question Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="font-bold text-gray-900">Q{index + 1}.</span>
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${
                        question.category === 'LLM' ? 'bg-blue-100 text-blue-800' :
                        question.category === 'LLMOps' ? 'bg-purple-100 text-purple-800' :
                        'bg-indigo-100 text-indigo-800'
                      }`}>
                        {question.category}
                      </span>
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${
                        question.difficulty === 'Easy' ? 'bg-green-100 text-green-800' :
                        question.difficulty === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {question.difficulty}
                      </span>
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      {question.question}
                    </h3>
                  </div>
                  <div className={`ml-4 px-4 py-2 rounded-lg font-bold ${
                    isCorrect ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                  }`}>
                    {isCorrect ? '✓ Correct' : '✗ Incorrect'}
                  </div>
                </div>

                {/* Options */}
                <div className="space-y-2 mb-4">
                  {question.options.map((option, optIndex) => {
                    const isUserAnswer = userAnswer === optIndex;
                    const isCorrectAnswer = optIndex === question.correctAnswer;

                    return (
                      <div
                        key={optIndex}
                        className={`p-3 rounded-lg border-2 ${
                          isCorrectAnswer
                            ? 'border-green-500 bg-green-50'
                            : isUserAnswer
                            ? 'border-red-500 bg-red-50'
                            : 'border-gray-200 bg-gray-50'
                        }`}
                      >
                        <div className="flex items-center">
                          {isCorrectAnswer && (
                            <span className="text-green-600 font-bold mr-2">✓</span>
                          )}
                          {isUserAnswer && !isCorrectAnswer && (
                            <span className="text-red-600 font-bold mr-2">✗</span>
                          )}
                          <span className={
                            isCorrectAnswer || isUserAnswer ? 'font-semibold' : ''
                          }>
                            {option}
                          </span>
                          {isUserAnswer && (
                            <span className="ml-2 text-sm text-gray-600">(Your answer)</span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Explanation */}
                <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
                  <div className="font-semibold text-blue-900 mb-1">Explanation:</div>
                  <div className="text-blue-800">{question.explanation}</div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Back to Top */}
        <div className="mt-8 text-center">
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="text-indigo-600 hover:text-indigo-700 font-semibold"
          >
            ↑ Back to Top
          </button>
        </div>
      </div>
    </div>
  );
}

export default Results;
