import React, { useState } from 'react';
import Home from './components/Home';
import Quiz from './components/Quiz';
import Results from './components/Results';
import Sessions from './components/Sessions';
import SessionDetail from './components/SessionDetail';
import { getRandomQuestions } from './data/questions';

function App() {
  const [currentView, setCurrentView] = useState('home');
  const [quizQuestions, setQuizQuestions] = useState([]);
  const [quizAnswers, setQuizAnswers] = useState([]);
  const [selectedSessionId, setSelectedSessionId] = useState(null);

  const handleStartQuiz = (selectedCategories) => {
    const questions = getRandomQuestions(30, selectedCategories);
    setQuizQuestions(questions);
    setCurrentView('quiz');
  };

  const handleQuizComplete = (answers) => {
    setQuizAnswers(answers);
    setCurrentView('results');
  };

  const handleBackToHome = () => {
    setCurrentView('home');
    setQuizQuestions([]);
    setQuizAnswers([]);
    setSelectedSessionId(null);
  };

  const handleViewSessions = () => {
    setCurrentView('sessions');
  };

  const handleViewSession = (sessionId) => {
    setSelectedSessionId(sessionId);
    setCurrentView('sessionDetail');
  };

  return (
    <>
      {currentView === 'home' && (
        <Home
          onStartQuiz={handleStartQuiz}
          onViewSessions={handleViewSessions}
        />
      )}

      {currentView === 'quiz' && (
        <Quiz
          questions={quizQuestions}
          onComplete={handleQuizComplete}
        />
      )}

      {currentView === 'results' && (
        <Results
          questions={quizQuestions}
          answers={quizAnswers}
          onBackToHome={handleBackToHome}
          onViewSessions={handleViewSessions}
        />
      )}

      {currentView === 'sessions' && (
        <Sessions
          onBackToHome={handleBackToHome}
          onViewSession={handleViewSession}
        />
      )}

      {currentView === 'sessionDetail' && (
        <SessionDetail
          sessionId={selectedSessionId}
          onBack={handleViewSessions}
        />
      )}
    </>
  );
}

export default App;
