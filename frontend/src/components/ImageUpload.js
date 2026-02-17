import React, { useCallback } from 'react';

import { Upload, X, Image as ImageIcon } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const ImageUpload = ({ onUpload }) => {
  const handleFileChange = (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) {
      onUpload(file);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) {
      onUpload(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-xl mx-auto"
    >
      <div
        className="relative border-2 border-dashed border-gray-600 rounded-2xl p-12 text-center hover:border-blue-500 transition-colors cursor-pointer bg-gray-800/50 backdrop-blur-sm"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <div className="flex flex-col items-center gap-4">
          <div className="p-4 bg-gray-700/50 rounded-full">
            <Upload className="w-8 h-8 text-blue-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white mb-2">Upload Car Image</h3>
            <p className="text-gray-400">Drag and drop or click to browse</p>
          </div>
          <div className="flex gap-2 text-sm text-gray-500">
            <span>JPG</span>
            <span>PNG</span>
            <span>WEBP</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default ImageUpload;
