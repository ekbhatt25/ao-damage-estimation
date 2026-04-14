import React from 'react';

import { Upload } from 'lucide-react';
import { motion } from 'framer-motion';

const ImageUpload = ({ onUpload }) => {
  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
      onUpload(files);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);
    if (files.length > 0) {
      onUpload(files);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-2xl mx-auto"
    >
      <div
        className="relative border-2 border-dashed border-gray-600 rounded-2xl p-12 text-center hover:border-blue-500 transition-colors cursor-pointer bg-gray-800/50 backdrop-blur-sm"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <div className="flex flex-col items-center gap-4">
          <div className="p-4 bg-gray-700/50 rounded-full">
            <Upload className="w-8 h-8 text-blue-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white mb-2">Upload Car Image(s)</h3>
            <p className="text-black">Drag and drop or click to browse</p>
          </div>
          <div className="flex gap-2 text-sm text-black">
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
