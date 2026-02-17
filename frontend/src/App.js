import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ImageUpload from './components/ImageUpload';
import LoadingOverlay from './components/LoadingOverlay';
import ResultsDisplay from './components/ResultsDisplay';
import './App.css';

function App() {
  const [appState, setAppState] = useState('idle'); // idle, analyzing, paused, complete, error
  const [image, setImage] = useState(null);
  const [results, setResults] = useState(null);
  const [progress, setProgress] = useState(0);

  // Simulation of analysis process
  useEffect(() => {
    let interval;
    if (appState === 'analyzing') {
      interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval);
            setAppState('complete');
            return 100;
          }
          return prev + 2; // increments every 100ms
        });
      }, 100);
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [appState]);

  const handleUpload = (file) => {
    setImage(URL.createObjectURL(file));
    setAppState('analyzing');
    setProgress(0);
  };

  const handlePause = () => {
    if (appState === 'analyzing') {
      setAppState('paused');
    } else if (appState === 'paused') {
      setAppState('analyzing');
    }
  };

  const handleReset = () => {
    setImage(null);
    setResults(null);
    setAppState('idle');
    setProgress(0);
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white selection:bg-blue-500/30">
      <div className="container mx-auto px-4 py-8 relative min-h-screen flex flex-col">

        {/* Header */}
        <header className="flex items-center justify-between py-6 mb-12 border-b border-gray-800/50">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
              <span className="font-bold text-white">AI</span>
            </div>
            <h1 className="text-xl font-bold tracking-tight">AutoDamage<span className="text-blue-400">Estimator</span></h1>
          </div>
          <nav className="hidden md:flex gap-6 text-sm font-medium text-gray-400">
            <a href="#" className="hover:text-white transition-colors">History</a>
            <a href="#" className="hover:text-white transition-colors">Settings</a>
          </nav>
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
                  <h2 className="text-4xl md:text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                    Instant Damage Analysis
                  </h2>
                  <p className="text-xl text-gray-400 max-w-2xl mx-auto">
                    Upload a photo of your vehicle damage to get an instant AI-powered repair cost estimation.
                  </p>
                </div>
                <ImageUpload onUpload={handleUpload} />
              </motion.div>
            )}

            {appState === 'complete' && (
              <ResultsDisplay results={results} onReset={handleReset} />
            )}
          </AnimatePresence>

          {/* Loading Overlay */}
          <AnimatePresence>
            {(appState === 'analyzing' || appState === 'paused') && (
              <LoadingOverlay
                isPaused={appState === 'paused'}
                onTogglePause={handlePause}
              />
            )}
          </AnimatePresence>
        </main>

        <footer className="mt-12 py-6 text-center text-sm text-gray-600 border-t border-gray-800/50">
          <p>&copy; 2024 AutoDamageEstimator. AI analysis may vary from actual repair costs.</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
