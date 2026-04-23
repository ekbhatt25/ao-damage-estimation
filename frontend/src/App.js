import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ImageUpload from './components/ImageUpload';
import LoadingOverlay from './components/LoadingOverlay';
import ResultsDisplay from './components/ResultsDisplay';
import { generatePDF, generateCSV } from './utils/reportGenerator';
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
  const [images, setImages] = useState([]);
  const [resultsList, setResultsList] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [state, setState] = useState('');
  const abortControllerRef = React.useRef(null);

  const handleUpload = async (files) => {
    const newImages = files.map(f => URL.createObjectURL(f));
    setImages(newImages);
    setAppState('analyzing');

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

    const fetchPromises = files.map(async (file) => {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('session_id', SESSION_ID);
      formData.append('state', state);

      try {
        const response = await fetch(`${API_URL}/detect`, {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        });
        return await response.json();
      } catch (error) {
        if (error.name === 'AbortError') {
          throw error;
        }
        console.error('Error:', error);
        return { error: "Failed to connect to backend. Is it running?" };
      }
    });

    try {
      const newResultsList = await Promise.all(fetchPromises);
      setResultsList(newResultsList);
      setCurrentIndex(0);
      setAppState('complete');
    } catch (error) {
      if (error.name === 'AbortError') {
        setAppState('idle');
      } else {
        console.error('Error in processing:', error);
        setResultsList([{ error: "Failed to connect to backend. Is it running?" }]);
        setAppState('complete');
      }
    }
  };

  const handleCancel = () => {
    abortControllerRef.current?.abort();
  };

  const handleReset = () => {
    abortControllerRef.current?.abort();
    setImages([]);
    setResultsList([]);
    setCurrentIndex(0);
    setAppState('idle');
  };

  const [pdfLoading, setPdfLoading] = useState(false);

  const handleDownloadPDF = async (indices) => {
    setPdfLoading(true);
    await generatePDF(resultsList, images, indices);
    setPdfLoading(false);
  };

  const handleDownloadCSV = (indices) => {
    generateCSV(resultsList, images, indices);
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

                <div className="flex justify-center mb-6">
                  <div className="flex items-center gap-3 bg-white/10 border border-gray-600 rounded-xl px-4 py-3">
                    <label className="text-black text-sm font-medium whitespace-nowrap">State</label>
                    <select
                      value={state}
                      onChange={e => setState(e.target.value)}
                      className="bg-transparent text-black text-sm outline-none border-b border-gray-500 focus:border-white pb-0.5 cursor-pointer"
                    >
                      <option value="">— select state —</option>
                      <option value="AL">Alabama</option>
                      <option value="AK">Alaska</option>
                      <option value="AZ">Arizona</option>
                      <option value="AR">Arkansas</option>
                      <option value="CA">California</option>
                      <option value="CO">Colorado</option>
                      <option value="CT">Connecticut</option>
                      <option value="DE">Delaware</option>
                      <option value="FL">Florida</option>
                      <option value="GA">Georgia</option>
                      <option value="HI">Hawaii</option>
                      <option value="ID">Idaho</option>
                      <option value="IL">Illinois</option>
                      <option value="IN">Indiana</option>
                      <option value="IA">Iowa</option>
                      <option value="KS">Kansas</option>
                      <option value="KY">Kentucky</option>
                      <option value="LA">Louisiana</option>
                      <option value="ME">Maine</option>
                      <option value="MD">Maryland</option>
                      <option value="MA">Massachusetts</option>
                      <option value="MI">Michigan</option>
                      <option value="MN">Minnesota</option>
                      <option value="MS">Mississippi</option>
                      <option value="MO">Missouri</option>
                      <option value="MT">Montana</option>
                      <option value="NE">Nebraska</option>
                      <option value="NV">Nevada</option>
                      <option value="NH">New Hampshire</option>
                      <option value="NJ">New Jersey</option>
                      <option value="NM">New Mexico</option>
                      <option value="NY">New York</option>
                      <option value="NC">North Carolina</option>
                      <option value="ND">North Dakota</option>
                      <option value="OH">Ohio</option>
                      <option value="OK">Oklahoma</option>
                      <option value="OR">Oregon</option>
                      <option value="PA">Pennsylvania</option>
                      <option value="RI">Rhode Island</option>
                      <option value="SC">South Carolina</option>
                      <option value="SD">South Dakota</option>
                      <option value="TN">Tennessee</option>
                      <option value="TX">Texas</option>
                      <option value="UT">Utah</option>
                      <option value="VT">Vermont</option>
                      <option value="VA">Virginia</option>
                      <option value="WA">Washington</option>
                      <option value="WV">West Virginia</option>
                      <option value="WI">Wisconsin</option>
                      <option value="WY">Wyoming</option>
                    </select>
                  </div>
                </div>
                <ImageUpload onUpload={handleUpload} />
              </motion.div>
            )}

            {appState === 'analyzing' && (
              <LoadingOverlay onCancel={handleCancel} />
            )}

            {appState === 'complete' && resultsList.length > 0 && (
              <div className="w-full">
                {resultsList.length > 1 && (
                  <div className="flex items-center justify-between mb-4">
                    <button
                      onClick={() => setCurrentIndex(c => Math.max(0, c - 1))}
                      disabled={currentIndex === 0}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:opacity-50 text-white rounded-lg font-medium transition-colors cursor-pointer disabled:cursor-not-allowed"
                    >
                      ← Previous
                    </button>
                    <p className="text-black font-semibold text-lg">Image {currentIndex + 1}</p>
                    <button
                      onClick={() => setCurrentIndex(c => Math.min(resultsList.length - 1, c + 1))}
                      disabled={currentIndex === resultsList.length - 1}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:opacity-50 text-white rounded-lg font-medium transition-colors cursor-pointer disabled:cursor-not-allowed"
                    >
                      Next →
                    </button>
                  </div>
                )}

                {/* Download bar */}
                <div className="flex justify-end gap-2 mb-3">
                  <button
                    onClick={() => handleDownloadCSV([currentIndex])}
                    className="px-3 py-1.5 text-xs font-medium bg-white/10 hover:bg-white/20 text-black border border-gray-600 rounded-lg transition-colors"
                  >
                    ↓ CSV{resultsList.length > 1 ? ` (Image ${currentIndex + 1})` : ''}
                  </button>
                  <button
                    onClick={() => handleDownloadPDF([currentIndex])}
                    disabled={pdfLoading}
                    className="px-3 py-1.5 text-xs font-medium bg-white/10 hover:bg-white/20 text-black border border-gray-600 rounded-lg transition-colors disabled:opacity-50"
                  >
                    {pdfLoading ? 'Generating…' : `↓ PDF${resultsList.length > 1 ? ` (Image ${currentIndex + 1})` : ''}`}
                  </button>
                  {resultsList.length > 1 && (
                    <>
                      <button
                        onClick={() => handleDownloadCSV(resultsList.map((_, i) => i))}
                        className="px-3 py-1.5 text-xs font-medium bg-white/10 hover:bg-white/20 text-black border border-gray-600 rounded-lg transition-colors"
                      >
                        ↓ CSV (All)
                      </button>
                      <button
                        onClick={() => handleDownloadPDF(resultsList.map((_, i) => i))}
                        disabled={pdfLoading}
                        className="px-3 py-1.5 text-xs font-medium bg-white/10 hover:bg-white/20 text-black border border-gray-600 rounded-lg transition-colors disabled:opacity-50"
                      >
                        {pdfLoading ? 'Generating…' : '↓ PDF (All)'}
                      </button>
                    </>
                  )}
                </div>

                <ResultsDisplay
                  key={currentIndex}
                  results={resultsList[currentIndex]}
                  imageUrl={images[currentIndex]}
                  onReset={handleReset}
                  sessionId={SESSION_ID}
                />
              </div>
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
