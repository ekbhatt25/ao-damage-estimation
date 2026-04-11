import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, AlertCircle, RefreshCw, ShieldCheck, ShieldAlert, TrendingUp, AlertTriangle } from 'lucide-react';

const severityDots = (severity) => {
    const level = { minor: 1, moderate: 2, severe: 3 }[severity?.toLowerCase()] ?? 0;
    return [1, 2, 3].map(i => (
        <div key={i} className={`w-2 h-2 rounded-full ${i <= level ? 'bg-yellow-500' : 'bg-gray-700'}`} />
    ));
};

const ResultsDisplay = ({ results, onReset }) => {
    if (!results) return null;

    const {
        detections = [],
        summary = {},
        error,
        stp_eligible,
        stp_reasoning,
        confidence_score,
        total_loss,
        explanation,
        cost,
        claim_id,
        fraud_flags = [],
    } = results;

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-2xl mx-auto bg-gray-800/50 backdrop-blur-md border border-gray-700 rounded-2xl overflow-hidden shadow-2xl"
        >
            {/* Header */}
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
                    : <p className="text-black ml-11">
                        {summary.total_damaged_parts ?? 0} damaged part{summary.total_damaged_parts !== 1 ? 's' : ''} detected
                      </p>
                }
            </div>

            <div className="p-8 space-y-6">

                {/* Fraud Flags */}
                {fraud_flags.length > 0 && (
                    <div className="p-4 rounded-xl border bg-orange-900/30 border-orange-700">
                        <div className="flex items-center gap-2 mb-2">
                            <AlertTriangle className="w-4 h-4 text-orange-400 flex-shrink-0" />
                            <p className="text-orange-400 font-bold text-sm">Image Fraud Signals Detected</p>
                        </div>
                        <ul className="space-y-1">
                            {fraud_flags.map((flag, i) => {
                                const [name, detail] = flag.split(' (');
                                return (
                                    <li key={i} className="text-xs text-gray-300">
                                        <span className="text-orange-300 font-medium">{name.replace(/_/g, ' ')}</span>
                                        {detail && <span className="text-gray-200"> — {detail.replace(')', '')}</span>}
                                    </li>
                                );
                            })}
                        </ul>
                    </div>
                )}

                {/* STP Decision Banner */}
                {stp_eligible != null && (
                    <div className={`p-4 rounded-xl border flex items-start gap-3 ${
                        total_loss
                            ? 'bg-red-900/30 border-red-700'
                            : stp_eligible
                                ? 'bg-green-900/30 border-green-700'
                                : 'bg-yellow-900/30 border-yellow-700'
                    }`}>
                        {total_loss
                            ? <TrendingUp className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
                            : stp_eligible
                                ? <ShieldCheck className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                                : <ShieldAlert className="w-5 h-5 text-yellow-400 mt-0.5 flex-shrink-0" />
                        }
                        <div>
                            <p className={`font-bold text-sm ${total_loss ? 'text-red-400' : stp_eligible ? 'text-green-400' : 'text-yellow-400'}`}>
                                {total_loss ? 'Total Loss' : stp_eligible ? 'Auto-Approved' : 'Adjuster Review Required'}
                            </p>
                            {stp_reasoning && (
                                <p className="text-gray-300 text-xs mt-1">{stp_reasoning}</p>
                            )}
                        </div>
                    </div>
                )}

                {/* Cost + Confidence row */}
                {(cost || confidence_score != null) && (
                    <div className="grid grid-cols-2 gap-4">
                        {cost?.total_cost_range && (
                            <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-700">
                                <label className="text-xs font-medium text-gray-400">Estimated Repair Cost</label>
                                <p className="text-white font-bold text-lg mt-1">
                                    ${cost.total_cost_range[0].toLocaleString()} – ${cost.total_cost_range[1].toLocaleString()}
                                </p>
                            </div>
                        )}
                        {confidence_score != null && (
                            <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-700">
                                <label className="text-xs font-medium text-gray-400">Claim Confidence</label>
                                <p className="text-white font-bold text-lg mt-1">
                                    {(confidence_score * 100).toFixed(0)}%
                                </p>
                            </div>
                        )}
                    </div>
                )}

                {/* LLM Explanation */}
                {explanation && (
                    <div className="p-4 bg-gray-900/50 rounded-xl border border-gray-700">
                        <label className="text-xs font-medium text-gray-400 block mb-2">AI Assessment</label>
                        <p className="text-gray-300 text-sm leading-relaxed">{explanation}</p>
                    </div>
                )}

                {/* Detections */}
                {detections.length === 0 && !error && (
                    <p className="text-black text-center py-4">No damage detected in this image.</p>
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
                        {/* Per-part cost */}
                        {cost?.damaged_parts?.[i]?.cost_range && (
                            <div className="space-y-1">
                                <label className="text-xs font-medium text-gray-400">Part Cost Range</label>
                                <p className="text-white text-sm">
                                    ${cost.damaged_parts[i].cost_range[0].toLocaleString()} – ${cost.damaged_parts[i].cost_range[1].toLocaleString()}
                                    <span className="text-gray-400 ml-2">({cost.damaged_parts[i].action})</span>
                                </p>
                            </div>
                        )}
                    </div>
                ))}

                {/* Claim ID */}
                {claim_id && (
                    <p className="text-black text-xs text-center">Claim ID: {claim_id}</p>
                )}

                <div className="pt-2 border-t border-gray-700">
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
