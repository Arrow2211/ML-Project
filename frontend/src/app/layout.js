import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], weight: ["300", "400", "500", "600", "700"] });

export const metadata = {
  title: "Multi-Hazard Risk Prediction System | India",
  description:
    "AI-powered disaster risk prediction for Indian cities — predicts Low, Medium, and High risk levels using environmental and geological data with explainable ML.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
