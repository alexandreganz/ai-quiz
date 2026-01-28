import React, { useEffect, useState } from 'react';
import { getSessionById } from '../services/storage';
import Results from './Results';

function SessionDetail({ sessionId, onBack }) {
  const [session, setSession] = useState(null);

  useEffect(() => {
    const sessionData = getSessionById(sessionId);
    setSession(sessionData);
  }, [sessionId]);

  if (!session) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
          <h2 className="text-2xl font-bold text-gray-900">Session not found</h2>
          <button
            onClick={onBack}
            className="mt-4 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-6 rounded-xl transition duration-200"
          >
            Back to Sessions
          </button>
        </div>
      </div>
    );
  }

  return (
    <Results
      questions={session.questions}
      answers={session.answers}
      onBackToHome={onBack}
      onViewSessions={onBack}
      sessionSaved={true}
    />
  );
}

export default SessionDetail;
