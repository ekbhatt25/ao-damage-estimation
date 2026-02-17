import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, AlertCircle, RefreshCw, ChevronRight } from 'lucide-react';

const ResultsDisplay = ({ results, onReset }) => {
    const mockData = {
        damageType: "Bumper Dent",
        severity: "Moderate",
        estimatedCost: "$450 - $800",
        recommendation: "Replace bumper cover and repaint."
    };

    const data = results;
    if (!data) return null;

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-2xl mx-auto bg-gray-800/50 backdrop-blur-md border border-gray-700 rounded-2xl overflow-hidden shadow-2xl"
        >
            <div className="p-8 border-b border-gray-700 bg-gradient-to-r from-blue-900/20 to-purple-900/20">
                <div className="flex items-center gap-3 mb-2">
                    <CheckCircle className="w-8 h-8 text-green-400" />
                    <h2 className="text-2xl font-bold text-white">Analysis Complete</h2>
                </div>
                <p className="text-gray-400 ml-11">Successfully moved to verification phase</p>
            </div>

            <div className="p-8 space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-400">Damage Type</label>
                        <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-700">
                            <span className="text-lg text-white font-medium">{data.damageType}</span>
                        </div>
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-400">Severity</label>
                        <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-700 flex items-center justify-between">
                            <span className="text-lg text-white font-medium">{data.severity}</span>
                            <div className="flex gap-1">
                                {[1, 2, 3, 4, 5].map(i => (
                                    <div key={i} className={`w-2 h-2 rounded-full ${i <= 3 ? 'bg-yellow-500' : 'bg-gray-700'}`} />
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-400">Estimated Repair Cost</label>
                    <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-700">
                        <span className="text-2xl text-green-400 font-bold">{data.estimatedCost}</span>
                    </div>
                </div>

                <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-400">AI Recommendation</label>
                    <div className="p-4 bg-blue-900/20 border border-blue-500/30 rounded-xl">
                        <p className="text-blue-200">{data.recommendation}</p>
                    </div>
                </div>

                <div className="pt-6 border-t border-gray-700">
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
