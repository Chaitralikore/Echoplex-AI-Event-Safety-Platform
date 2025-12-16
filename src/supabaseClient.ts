import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://pbkqezvgflfsbsqarcqm.supabase.co ';      // from Settings → Data API → Project URL
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBia3FlenZnZmxmc2JzcWFyY3FtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ2NjAwMDAsImV4cCI6MjA4MDIzNjAwMH0.XdaaS6Mc-EL19lUe72QVDwbW7wgFDnbxrORHuuPVaRw  ';     // the anon public key you just copied

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
