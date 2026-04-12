import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ImageUpload from './components/ImageUpload';
import LoadingOverlay from './components/LoadingOverlay';
import ResultsDisplay from './components/ResultsDisplay';
import './App.css';
import aoLogo from './Auto_Owners_Logo_full_circle.jpg';
import aoTextLogo from './ao-text-logo.jpg';

function getSessionId() {
  let id = localStorage.getItem('ao_session_id');
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem('ao_session_id', id);
  }
  return id;
}

const SESSION_ID = getSessionId();

function App() {
  const [appState, setAppState] = useState('idle');
  const [image, setImage] = useState(null);
  const [results, setResults] = useState(null);
  const abortControllerRef = React.useRef(null);

  const handleUpload = async (file) => {
    setImage(URL.createObjectURL(file));
    setAppState('analyzing');

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const formData = new FormData();
    formData.append('image', file);
    formData.append('session_id', SESSION_ID);

    try {
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/detect`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      const data = await response.json();
      setResults(data);
      setAppState('complete');

    } catch (error) {
      if (error.name === 'AbortError') {
        setAppState('idle');
      } else {
        console.error('Error:', error);
        setResults({ error: "Failed to connect to backend. Is it running?" });
        setAppState('complete');
      }
    }
  };

  const handleCancel = () => {
    abortControllerRef.current?.abort();
  };

  const handleReset = () => {
    abortControllerRef.current?.abort();
    setImage(null);
    setResults(null);
    setAppState('idle');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#084477] via-[#0a5694] to-[#084477] text-white selection:bg-[#084477]/30">
      <div className="container mx-auto px-4 py-0 relative min-h-screen flex flex-col">

        {/* Header */}
        <header className="flex items-center justify-center py-2 mb-16 border-b border-gray-800/50 w-full">
          <div className="flex items-center justify-center gap-4 w-full px-8">
            <img src={aoLogo} alt="Auto-Owners logo" className="h-28 w-28 object-contain flex-shrink-0" />
            <img src={aoTextLogo} alt="Auto-Owners Insurance" className="h-28 object-contain" />
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-grow flex flex-col items-center justify-center relative z-10">
          <AnimatePresence mode="wait">
            {appState === 'idle' && (
              <motion.div
                key="upload"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="w-full"
              >
                <div className="text-center mb-12">
                  <h2 className="text-3xl font-bold mb-6 text-black whitespace-nowrap">
                    Vehicle Damage Estimator
                  </h2>
                  <p className="text-xl text-black max-w-2xl mx-auto">
                    Upload a photo of the damaged vehicle for an instant AI-powered repair cost estimation.
                  </p>
                </div>

                <ImageUpload onUpload={handleUpload} />
              </motion.div>
            )}

            {appState === 'analyzing' && (
              <LoadingOverlay onCancel={handleCancel} />
            )}

            {appState === 'complete' && (
              <ResultsDisplay results={results} imageUrl={image} onReset={handleReset} sessionId={SESSION_ID} />
            )}
          </AnimatePresence>
        </main>

        <footer className="mt-12 py-6 text-center text-sm text-black border-t border-gray-800/50">
          <p>&copy; 2026 Auto-Owners Vehicle Damage Estimator.</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
