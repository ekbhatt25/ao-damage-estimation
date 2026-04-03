import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ImageUpload from './components/ImageUpload';
import LoadingOverlay from './components/LoadingOverlay';
import ResultsDisplay from './components/ResultsDisplay';
import './App.css';

function App() {
  const [appState, setAppState] = useState('idle');
  const [image, setImage] = useState(null);
  const [results, setResults] = useState(null);

  const handleUpload = async (file) => {
    setImage(URL.createObjectURL(file));
    setAppState('analyzing');

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResults(data);
      setAppState('complete');

    } catch (error) {
      console.error('Error:', error);
      setResults({ error: "Failed to connect to backend. Is it running?" });
      setAppState('complete');
    }
  };

  const handleReset = () => {
    setImage(null);
    setResults(null);
    setAppState('idle');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#084477] via-[#0a5694] to-[#084477] text-white selection:bg-[#084477]/30">
      <div className="container mx-auto px-4 py-8 relative min-h-screen flex flex-col">

        {/* Header */}
        <header className="flex items-center justify-center py-6 mb-12 border-b border-gray-800/50">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-[#084477] rounded-lg flex items-center justify-center">
              <span className="font-bold text-white">AO</span>
            </div>
            <h1 className="text-xl font-bold tracking-tight">Auto-Owners<span className="text-[#0a5694]"> Damage Estimator</span></h1>
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
                  <h2 className="text-4xl md:text-5xl font-bold mb-6 text-black">
                    Damaged Car Analysis
                  </h2>
                  <p className="text-xl text-black max-w-2xl mx-auto">
                    Upload a photo of the damaged vehicle for an instant AI-powered repair cost estimation.
                  </p>
                </div>
                <ImageUpload onUpload={handleUpload} />
              </motion.div>
            )}

            {appState === 'analyzing' && (
              <LoadingOverlay isPaused={false} onTogglePause={() => {}} />
            )}

            {appState === 'complete' && (
              <ResultsDisplay results={results} onReset={handleReset} />
            )}
          </AnimatePresence>
        </main>

        <footer className="mt-12 py-6 text-center text-sm text-gray-400 border-t border-gray-800/50">
          <p>&copy; 2026 Auto-Owners Damage Estimator. Repair cost & severity estimations are pending.</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
