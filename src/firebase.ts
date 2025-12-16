


// src/firebase.ts
import { initializeApp } from "firebase/app";
import { getDatabase } from "firebase/database";
import { getStorage } from 'firebase/storage';

// 1) PASTE YOUR CONFIG OBJECT FROM THE FIREBASE CONSOLE HERE
const firebaseConfig = {
  apiKey: "AIzaSyBr8xw7LQasYKej6fInOXX_6ovB2N1mZY",
  authDomain: "echoplex-final.firebaseapp.com",
  databaseURL: "https://echoplex-final-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "echoplex-final",
  storageBucket: "echoplex-final.firebasestorage.app",
  messagingSenderId: "610329800314",
  appId: "1:610329800314:web:7b2f1767be8cd83c8e55c3",
  measurementId: "G-QXJ2E2KHWS"
};

// 2) INITIALIZE FIREBASE APP
const app = initializeApp(firebaseConfig);

// 3) GET REALTIME DATABASE INSTANCE
export const db = getDatabase(app);
export const storage = getStorage(app);
