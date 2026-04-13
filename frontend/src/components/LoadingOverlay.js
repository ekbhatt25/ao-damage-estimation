import React from 'react';
import { motion } from 'framer-motion';
import { X } from 'lucide-react';

const LoadingOverlay = ({ onCancel }) => {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/80 backdrop-blur-md"
        >
            <div className="text-center space-y-8">
                <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    className="w-24 h-24 mx-auto border-4 border-blue-500/30 border-t-blue-500 rounded-full"
                />

                <div className="space-y-2">
                    <h3 className="text-2xl font-bold text-white">Analyzing Damage...</h3>
                    <p className="text-gray-400">Our AI is scanning the vehicle</p>
                </div>

                <button
                    onClick={onCancel}
                    className="flex items-center gap-2 px-6 py-3 bg-white/10 hover:bg-white/20 rounded-full text-white transition-all mx-auto"
                >
                    <X className="w-5 h-5" />
                    <span>Cancel Analysis</span>
                </button>
            </div>
        </motion.div>
    );
};

export default LoadingOverlay;
