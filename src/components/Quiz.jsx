import React, { useState } from 'react';

function Quiz({ questions, onComplete, onBackToHome }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState(Array(questions.length).fill(null));
  const [selectedOption, setSelectedOption] = useState(null);
  const [showFeedback, setShowFeedback] = useState(false);

  const currentQuestion = questions[currentIndex];
  const isLastQuestion = currentIndex === questions.length - 1;
  const progress = ((currentIndex + 1) / questions.length) * 100;
  const isCorrect = selectedOption === currentQuestion.correctAnswer;

  const handleOptionSelect = (optionIndex) => {
    if (!showFeedback) {
      setSelectedOption(optionIndex);
    }
  };

  const handleCheckAnswer = () => {
    if (selectedOption !== null) {
      setShowFeedback(true);
      const newAnswers = [...answers];
      newAnswers[currentIndex] = selectedOption;
      setAnswers(newAnswers);
    }
  };

  const handleNext = () => {
    if (isLastQuestion) {
      onComplete(answers);
    } else {
      setCurrentIndex(currentIndex + 1);
      setSelectedOption(answers[currentIndex + 1]);
      setShowFeedback(false);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
      setSelectedOption(answers[currentIndex - 1]);
      setShowFeedback(false);
    }
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'Easy':
        return 'bg-green-100 text-green-800';
      case 'Medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'Hard':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getCategoryColor = (category) => {
    switch (category) {
      case 'LLM':
        return 'bg-blue-100 text-blue-800';
      case 'LLMOps':
        return 'bg-purple-100 text-purple-800';
      case 'GenAI':
        return 'bg-indigo-100 text-indigo-800';
      case 'Forecasting':
        return 'bg-orange-100 text-orange-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getOptionStyle = (index) => {
    if (!showFeedback) {
      return selectedOption === index
        ? 'border-indigo-600 bg-indigo-50 shadow-md'
        : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50';
    }

    // Show feedback
    const isThisCorrect = index === currentQuestion.correctAnswer;
    const isUserSelection = index === selectedOption;

    if (isThisCorrect) {
      return 'border-green-500 bg-green-50 shadow-md';
    }
    if (isUserSelection && !isThisCorrect) {
      return 'border-red-500 bg-red-50 shadow-md';
    }
    return 'border-gray-200 bg-gray-50 opacity-60';
  };

  const getOptionIcon = (index) => {
    if (!showFeedback) {
      return (
        <div
          className={`w-6 h-6 rounded-full border-2 mr-3 flex items-center justify-center ${
            selectedOption === index
              ? 'border-indigo-600 bg-indigo-600'
              : 'border-gray-300'
          }`}
        >
          {selectedOption === index && (
            <div className="w-2 h-2 bg-white rounded-full"></div>
          )}
        </div>
      );
    }

    const isThisCorrect = index === currentQuestion.correctAnswer;
    const isUserSelection = index === selectedOption;

    if (isThisCorrect) {
      return (
        <div className="w-6 h-6 rounded-full bg-green-500 mr-3 flex items-center justify-center text-white font-bold">
          ✓
        </div>
      );
    }
    if (isUserSelection && !isThisCorrect) {
      return (
        <div className="w-6 h-6 rounded-full bg-red-500 mr-3 flex items-center justify-center text-white font-bold">
          ✗
        </div>
      );
    }
    return (
      <div className="w-6 h-6 rounded-full border-2 border-gray-300 mr-3"></div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4 py-8">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              <button
                onClick={onBackToHome}
                className="text-indigo-600 hover:text-indigo-800 font-semibold transition duration-200 flex items-center gap-1"
                title="Back to Home"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                Home
              </button>
              <div>
                <h2 className="text-2xl font-bold text-gray-900">
                  Question {currentIndex + 1} of {questions.length}
                </h2>
              </div>
            </div>
            <div className="flex gap-2">
              <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getCategoryColor(currentQuestion.category)}`}>
                {currentQuestion.category}
              </span>
              <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getDifficultyColor(currentQuestion.difficulty)}`}>
                {currentQuestion.difficulty}
              </span>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>

        {/* Question Card */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">
            {currentQuestion.question}
          </h3>

          {/* Options */}
          <div className="space-y-3 mb-6">
            {currentQuestion.options.map((option, index) => (
              <button
                key={index}
                onClick={() => handleOptionSelect(index)}
                disabled={showFeedback}
                className={`w-full text-left p-4 rounded-xl border-2 transition-all duration-200 ${getOptionStyle(index)} ${
                  showFeedback ? 'cursor-default' : 'cursor-pointer'
                }`}
              >
                <div className="flex items-center">
                  {getOptionIcon(index)}
                  <span className="text-gray-900">{option}</span>
                </div>
              </button>
            ))}
          </div>

          {/* Feedback Section */}
          {showFeedback && (
            <div className={`mt-6 p-6 rounded-xl border-2 ${
              isCorrect
                ? 'bg-green-50 border-green-300'
                : 'bg-red-50 border-red-300'
            }`}>
              <div className="flex items-center mb-3">
                <div className={`text-2xl font-bold ${
                  isCorrect ? 'text-green-700' : 'text-red-700'
                }`}>
                  {isCorrect ? '✓ Correct!' : '✗ Incorrect'}
                </div>
              </div>

              {!isCorrect && (
                <div className="mb-3 text-gray-800">
                  <span className="font-semibold">Correct answer: </span>
                  {currentQuestion.options[currentQuestion.correctAnswer]}
                </div>
              )}

              <div className="bg-white bg-opacity-70 p-4 rounded-lg">
                <div className="font-semibold text-gray-900 mb-2">Explanation:</div>
                <div className="text-gray-800">{currentQuestion.explanation}</div>
              </div>
            </div>
          )}
        </div>

        {/* Navigation Buttons */}
        <div className="flex gap-4">
          <button
            onClick={handlePrevious}
            disabled={currentIndex === 0}
            className={`px-6 py-3 rounded-xl font-semibold transition duration-200 ${
              currentIndex === 0
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-white text-indigo-600 border-2 border-indigo-600 hover:bg-indigo-50'
            }`}
          >
            Previous
          </button>

          {!showFeedback ? (
            <button
              onClick={handleCheckAnswer}
              disabled={selectedOption === null}
              className={`flex-1 py-3 rounded-xl font-semibold transition duration-200 ${
                selectedOption === null
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg hover:shadow-xl'
              }`}
            >
              Check Answer
            </button>
          ) : (
            <button
              onClick={handleNext}
              className="flex-1 py-3 rounded-xl font-semibold transition duration-200 bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg hover:shadow-xl"
            >
              {isLastQuestion ? 'View Results' : 'Next Question'}
            </button>
          )}
        </div>

        {/* Question Overview */}
        <div className="mt-6 bg-white rounded-xl shadow p-4">
          <div className="text-sm text-gray-600 mb-2">Progress</div>
          <div className="flex flex-wrap gap-2">
            {questions.map((_, index) => (
              <div
                key={index}
                className={`w-8 h-8 rounded flex items-center justify-center text-sm font-semibold ${
                  index === currentIndex
                    ? 'bg-indigo-600 text-white'
                    : answers[index] !== null
                    ? 'bg-green-100 text-green-800'
                    : 'bg-gray-100 text-gray-600'
                }`}
              >
                {index + 1}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Quiz;
