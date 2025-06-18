// File: LandingPage.jsx

import React from 'react';
import { motion } from 'framer-motion';

const features = [
  {
    icon: '🧩',
    title: 'Match by Vibes',
    description: 'Not just preferences — personality, habits, and memes too.',
  },
  {
    icon: '🔐',
    title: 'Safe & Verified',
    description: 'Every profile is human-checked. Your safety is our job.',
  },
  {
    icon: '🛋',
    title: 'Move-In Ready Tools',
    description: 'Contracts, chats, and checklists — all in one platform.',
  },
];

export default function LandingPage() {
  return (
    <div className="relative min-h-screen bg-gradient-to-br from-sky-100 via-white to-amber-100 text-gray-800 overflow-x-hidden">
      {/* Hero Section */}
      <header className="flex flex-col items-center justify-center text-center px-4 py-20">
        <motion.h1
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
          className="text-5xl md:text-6xl font-bold mb-4"
        >
          Find Your Flatmate, Not Your Soulmate.
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 1 }}
          className="text-xl md:text-2xl mb-8 max-w-xl text-gray-600"
        >
          A cozy chaos of vibes, rent, and roommates — made for Gen Z by Gen Z 🧃
        </motion.p>

        <motion.a
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          href="#join"
          className="bg-indigo-600 text-white px-6 py-3 rounded-full shadow hover:bg-indigo-500 transition-all"
        >
          Get Started
        </motion.a>
      </header>

      {/* Video Section */}
      <motion.section
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        transition={{ duration: 1 }}
        viewport={{ once: true }}
        className="flex justify-center p-4 md:p-12"
        id="video"
      >
        <video
          src="/videos/roomtour.mp4"
          autoPlay
          loop
          muted
          playsInline
          className="rounded-xl shadow-lg w-full max-w-4xl border-4 border-white"
        />
      </motion.section>

      {/* Features */}
      <section className="grid md:grid-cols-3 gap-6 px-6 py-12 text-center bg-white" id="features">
        {features.map((feature, index) => (
          <motion.div
            key={index}
            className="p-6 rounded-lg hover:shadow-lg transition bg-sky-50"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.3, duration: 0.6 }}
            viewport={{ once: true }}
          >
            <div className="text-3xl">{feature.icon}</div>
            <h3 className="text-xl font-semibold mt-4">{feature.title}</h3>
            <p className="text-gray-600 mt-1">{feature.description}</p>
          </motion.div>
        ))}
      </section>

      {/* CTA */}
      <motion.section
        initial={{ scale: 0.9, opacity: 0 }}
        whileInView={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.3 }}
        viewport={{ once: true }}
        className="bg-indigo-600 text-white py-16 px-6 text-center" id="join"
      >
        <h2 className="text-3xl md:text-4xl font-bold mb-4">Ready to find your perfect match?</h2>
        <p className="text-lg mb-8">Roommate-wise, of course 😌</p>
        <motion.a
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          href="/signup"
          className="bg-white text-indigo-600 font-semibold px-8 py-3 rounded-full hover:bg-gray-100 transition"
        >
          Sign Up Now
        </motion.a>
      </motion.section>

      {/* Footer */}
      <footer className="py-6 text-center text-gray-500 text-sm">
        © 2025 CoBae. No awkward interviews, just vibes.
      </footer>
    </div>
  );
}