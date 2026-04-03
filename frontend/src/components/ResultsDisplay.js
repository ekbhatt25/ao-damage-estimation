import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';

const severityDots = (severity) => {
    const level = { minor: 1, moderate: 2, severe: 3 }[severity?.toLowerCase()] ?? 0;
    return [1, 2, 3].map(i => (
        <div key={i} className={`w-2 h-2 rounded-full ${i <= level ? 'bg-yellow-500' : 'bg-gray-700'}`} />
    ));
};

const ResultsDisplay = ({ results, onReset }) => {
    if (!results) return null;

    const { detections = [], summary = {}, error } = results;

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-2xl mx-auto bg-gray-800/50 backdrop-blur-md border border-gray-700 rounded-2xl overflow-hidden shadow-2xl"
        >
            <div className="p-8 border-b border-gray-700 bg-gradient-to-r from-blue-900/20 to-purple-900/20">
                <div className="flex items-center gap-3 mb-2">
                    {error
                        ? <AlertCircle className="w-8 h-8 text-red-400" />
                        : <CheckCircle className="w-8 h-8 text-green-400" />
                    }
                    <h2 className="text-2xl font-bold text-white">Analysis Complete</h2>
                </div>
                {error
                    ? <p className="text-red-400 ml-11">{error}</p>
                    : <p className="text-gray-400 ml-11">
                        {summary.total_damaged_parts ?? 0} damaged part{summary.total_damaged_parts !== 1 ? 's' : ''} detected
                      </p>
                }
            </div>

            <div className="p-8 space-y-4">
                {detections.length === 0 && !error && (
                    <p className="text-gray-400 text-center py-4">No damage detected in this image.</p>
                )}

                {detections.map((det, i) => (
                    <div key={i} className="p-4 bg-gray-900/50 rounded-xl border border-gray-700 space-y-3">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-gray-400">Part</label>
                                <p className="text-white font-medium">{det.part}</p>
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-gray-400">Damage Type</label>
                                <p className="text-white font-medium">{det.damage_type}</p>
                            </div>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-gray-400">Severity</label>
                                <div className="flex items-center gap-2">
                                    <span className="text-white capitalize">{det.severity ?? '—'}</span>
                                    <div className="flex gap-1">{severityDots(det.severity)}</div>
                                </div>
                            </div>
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-gray-400">Confidence</label>
                                <p className="text-white">{det.confidence != null ? `${(det.confidence * 100).toFixed(0)}%` : '—'}</p>
                            </div>
                        </div>
                    </div>
                ))}

                <div className="pt-4 border-t border-gray-700">
                    <button
                        onClick={onReset}
                        className="w-full py-4 bg-white text-gray-900 rounded-xl font-bold hover:bg-gray-200 transition-colors flex items-center justify-center gap-2"
                    >
                        <RefreshCw className="w-5 h-5" />
                        Analyze Another Image
                    </button>
                </div>
            </div>
        </motion.div>
    );
};

export default ResultsDisplay;
