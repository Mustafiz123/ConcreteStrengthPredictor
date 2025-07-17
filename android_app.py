from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton, MDIconButton
from kivymd.uix.slider import MDSlider
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.navigationdrawer import MDNavigationDrawer, MDNavigationDrawerMenu
from kivymd.uix.list import OneLineListItem
from kivymd.uix.bottomnavigation import MDBottomNavigation, MDBottomNavigationItem

import numpy as np
import pandas as pd
import json
import os
from utilities import normalize_features, eval_cost, eval_gradient, predict_strength
from ml_model import ConcreteStrengthModel
from data_loader import load_concrete_data, get_feature_names, get_feature_ranges

class ConcreteStrengthApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = ConcreteStrengthModel()
        self.feature_ranges = get_feature_ranges()
        self.feature_names = get_feature_names()
        self.model_trained = False
        self.df = None
        
    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"
        
        # Create main screen
        screen = MDScreen()
        
        # Create main layout
        main_layout = MDBoxLayout(orientation='vertical')
        
        # Add top app bar
        toolbar = MDTopAppBar(
            title="Concrete Strength Predictor",
            elevation=2,
            md_bg_color=self.theme_cls.primary_color
        )
        main_layout.add_widget(toolbar)
        
        # Create bottom navigation
        self.bottom_nav = MDBottomNavigation()
        
        # Data tab
        data_tab = MDBottomNavigationItem(
            name='data',
            text='Data',
            icon='database'
        )
        data_tab.add_widget(self.create_data_tab())
        self.bottom_nav.add_widget(data_tab)
        
        # Training tab
        training_tab = MDBottomNavigationItem(
            name='training',
            text='Train',
            icon='brain'
        )
        training_tab.add_widget(self.create_training_tab())
        self.bottom_nav.add_widget(training_tab)
        
        # Prediction tab
        prediction_tab = MDBottomNavigationItem(
            name='prediction',
            text='Predict',
            icon='crystal-ball'
        )
        prediction_tab.add_widget(self.create_prediction_tab())
        self.bottom_nav.add_widget(prediction_tab)
        
        # Performance tab
        performance_tab = MDBottomNavigationItem(
            name='performance',
            text='Results',
            icon='chart-line'
        )
        performance_tab.add_widget(self.create_performance_tab())
        self.bottom_nav.add_widget(performance_tab)
        
        main_layout.add_widget(self.bottom_nav)
        screen.add_widget(main_layout)
        
        # Load data on start
        Clock.schedule_once(self.load_data, 0.1)
        
        return screen
    
    def create_data_tab(self):
        layout = MDBoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))
        
        # Title
        title = MDLabel(
            text="Dataset Overview",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(40),
            font_style="H5"
        )
        layout.add_widget(title)
        
        # Dataset info card
        info_card = MDCard(
            padding=dp(15),
            size_hint_y=None,
            height=dp(200),
            elevation=2
        )
        info_layout = MDBoxLayout(orientation='vertical')
        
        self.data_info_label = MDLabel(
            text="Loading dataset...",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(150)
        )
        info_layout.add_widget(self.data_info_label)
        info_card.add_widget(info_layout)
        layout.add_widget(info_card)
        
        # Features description card
        features_card = MDCard(
            padding=dp(15),
            size_hint_y=None,
            height=dp(300),
            elevation=2
        )
        features_layout = MDBoxLayout(orientation='vertical')
        
        features_title = MDLabel(
            text="Features",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
            font_style="H6"
        )
        features_layout.add_widget(features_title)
        
        # Create scrollable features list
        scroll = ScrollView()
        features_list = MDBoxLayout(orientation='vertical', size_hint_y=None)
        features_list.bind(minimum_height=features_list.setter('height'))
        
        for feature, description in self.feature_names.items():
            feature_item = MDLabel(
                text=f"• {feature}: {description}",
                theme_text_color="Secondary",
                size_hint_y=None,
                height=dp(30),
                text_size=(None, None)
            )
            features_list.add_widget(feature_item)
        
        scroll.add_widget(features_list)
        features_layout.add_widget(scroll)
        features_card.add_widget(features_layout)
        layout.add_widget(features_card)
        
        return layout
    
    def create_training_tab(self):
        layout = MDBoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))
        
        # Title
        title = MDLabel(
            text="Model Training",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(40),
            font_style="H5"
        )
        layout.add_widget(title)
        
        # Training parameters card
        params_card = MDCard(
            padding=dp(15),
            size_hint_y=None,
            height=dp(300),
            elevation=2
        )
        params_layout = MDBoxLayout(orientation='vertical', spacing=dp(10))
        
        params_title = MDLabel(
            text="Training Parameters",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
            font_style="H6"
        )
        params_layout.add_widget(params_title)
        
        # Learning rate slider
        lr_label = MDLabel(text="Learning Rate: 0.01", size_hint_y=None, height=dp(30))
        params_layout.add_widget(lr_label)
        
        self.lr_slider = MDSlider(
            min=0.001, max=0.1, value=0.01, step=0.001,
            size_hint_y=None, height=dp(30)
        )
        self.lr_slider.bind(value=lambda x, value: setattr(lr_label, 'text', f"Learning Rate: {value:.3f}"))
        params_layout.add_widget(self.lr_slider)
        
        # Iterations slider
        iter_label = MDLabel(text="Iterations: 1000", size_hint_y=None, height=dp(30))
        params_layout.add_widget(iter_label)
        
        self.iter_slider = MDSlider(
            min=100, max=5000, value=1000, step=100,
            size_hint_y=None, height=dp(30)
        )
        self.iter_slider.bind(value=lambda x, value: setattr(iter_label, 'text', f"Iterations: {int(value)}"))
        params_layout.add_widget(self.iter_slider)
        
        # Test size slider
        test_label = MDLabel(text="Test Size: 0.2", size_hint_y=None, height=dp(30))
        params_layout.add_widget(test_label)
        
        self.test_slider = MDSlider(
            min=0.1, max=0.4, value=0.2, step=0.05,
            size_hint_y=None, height=dp(30)
        )
        self.test_slider.bind(value=lambda x, value: setattr(test_label, 'text', f"Test Size: {value:.2f}"))
        params_layout.add_widget(self.test_slider)
        
        params_card.add_widget(params_layout)
        layout.add_widget(params_card)
        
        # Train button
        self.train_button = MDRaisedButton(
            text="Start Training",
            size_hint_y=None,
            height=dp(50),
            on_release=self.train_model
        )
        layout.add_widget(self.train_button)
        
        # Training results card
        self.results_card = MDCard(
            padding=dp(15),
            size_hint_y=None,
            height=dp(150),
            elevation=2
        )
        results_layout = MDBoxLayout(orientation='vertical')
        
        self.training_status = MDLabel(
            text="Ready to train model",
            theme_text_color="Secondary",
            size_hint_y=None,
            height=dp(120)
        )
        results_layout.add_widget(self.training_status)
        self.results_card.add_widget(results_layout)
        layout.add_widget(self.results_card)
        
        return layout
    
    def create_prediction_tab(self):
        layout = MDBoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))
        
        # Title
        title = MDLabel(
            text="Concrete Strength Prediction",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(40),
            font_style="H5"
        )
        layout.add_widget(title)
        
        # Create scrollable content
        scroll = ScrollView()
        scroll_layout = MDBoxLayout(orientation='vertical', spacing=dp(10), size_hint_y=None)
        scroll_layout.bind(minimum_height=scroll_layout.setter('height'))
        
        # Input sliders
        self.input_sliders = {}
        
        for feature, (min_val, max_val) in self.feature_ranges.items():
            # Feature card
            feature_card = MDCard(
                padding=dp(10),
                size_hint_y=None,
                height=dp(100),
                elevation=1
            )
            feature_layout = MDBoxLayout(orientation='vertical', spacing=dp(5))
            
            # Feature label
            default_val = (min_val + max_val) / 2
            feature_label = MDLabel(
                text=f"{self.feature_names[feature]}: {default_val:.1f}",
                size_hint_y=None,
                height=dp(30),
                font_style="Subtitle1"
            )
            feature_layout.add_widget(feature_label)
            
            # Feature slider
            feature_slider = MDSlider(
                min=min_val, max=max_val, value=default_val,
                size_hint_y=None, height=dp(30)
            )
            
            # Bind slider to update label
            def update_label(slider, value, label=feature_label, name=self.feature_names[feature]):
                label.text = f"{name}: {value:.1f}"
            
            feature_slider.bind(value=update_label)
            feature_layout.add_widget(feature_slider)
            
            # Range info
            range_label = MDLabel(
                text=f"Range: {min_val} - {max_val}",
                size_hint_y=None,
                height=dp(25),
                theme_text_color="Secondary",
                font_style="Caption"
            )
            feature_layout.add_widget(range_label)
            
            feature_card.add_widget(feature_layout)
            scroll_layout.add_widget(feature_card)
            
            self.input_sliders[feature] = feature_slider
        
        scroll.add_widget(scroll_layout)
        layout.add_widget(scroll)
        
        # Predict button
        self.predict_button = MDRaisedButton(
            text="Predict Strength",
            size_hint_y=None,
            height=dp(50),
            on_release=self.make_prediction
        )
        layout.add_widget(self.predict_button)
        
        # Prediction result card
        self.prediction_card = MDCard(
            padding=dp(15),
            size_hint_y=None,
            height=dp(100),
            elevation=2
        )
        prediction_layout = MDBoxLayout(orientation='vertical')
        
        self.prediction_result = MDLabel(
            text="Train model first to make predictions",
            theme_text_color="Secondary",
            size_hint_y=None,
            height=dp(70),
            font_style="H6"
        )
        prediction_layout.add_widget(self.prediction_result)
        self.prediction_card.add_widget(prediction_layout)
        layout.add_widget(self.prediction_card)
        
        return layout
    
    def create_performance_tab(self):
        layout = MDBoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))
        
        # Title
        title = MDLabel(
            text="Model Performance",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(40),
            font_style="H5"
        )
        layout.add_widget(title)
        
        # Performance metrics card
        self.performance_card = MDCard(
            padding=dp(15),
            size_hint_y=None,
            height=dp(200),
            elevation=2
        )
        performance_layout = MDBoxLayout(orientation='vertical')
        
        self.performance_metrics = MDLabel(
            text="Train model to see performance metrics",
            theme_text_color="Secondary",
            size_hint_y=None,
            height=dp(170)
        )
        performance_layout.add_widget(self.performance_metrics)
        self.performance_card.add_widget(performance_layout)
        layout.add_widget(self.performance_card)
        
        # Model save/load buttons
        buttons_layout = MDBoxLayout(
            orientation='horizontal',
            spacing=dp(10),
            size_hint_y=None,
            height=dp(50)
        )
        
        save_button = MDRaisedButton(
            text="Save Model",
            size_hint_x=0.5,
            on_release=self.save_model
        )
        buttons_layout.add_widget(save_button)
        
        load_button = MDRaisedButton(
            text="Load Model",
            size_hint_x=0.5,
            on_release=self.load_model
        )
        buttons_layout.add_widget(load_button)
        
        layout.add_widget(buttons_layout)
        
        return layout
    
    def load_data(self, dt):
        """Load the concrete dataset"""
        try:
            self.df = load_concrete_data()
            
            # Update data info
            info_text = f"""Dataset Information:
• Samples: {len(self.df)}
• Features: {len(self.df.columns) - 1}
• Target: Concrete Strength (MPa)

Statistics:
• Min Strength: {self.df['ConcreteStrength'].min():.1f} MPa
• Max Strength: {self.df['ConcreteStrength'].max():.1f} MPa
• Mean Strength: {self.df['ConcreteStrength'].mean():.1f} MPa"""
            
            self.data_info_label.text = info_text
            
        except Exception as e:
            self.data_info_label.text = f"Error loading data: {str(e)}"
    
    def train_model(self, button):
        """Train the machine learning model"""
        if self.df is None:
            self.show_popup("Error", "Please load data first")
            return
        
        # Disable button during training
        self.train_button.disabled = True
        self.train_button.text = "Training..."
        self.training_status.text = "Training model, please wait..."
        
        # Schedule training in next frame to update UI
        Clock.schedule_once(self._do_training, 0.1)
    
    def _do_training(self, dt):
        """Actual training function"""
        try:
            X, y = self.model.prepare_data(self.df)
            
            # Get parameters from sliders
            learning_rate = self.lr_slider.value
            num_iterations = int(self.iter_slider.value)
            test_size = self.test_slider.value
            
            # Train the model
            results = self.model.train(
                X, y,
                learning_rate=learning_rate,
                num_iterations=num_iterations,
                test_size=test_size
            )
            
            self.model_trained = True
            
            # Update training status
            status_text = f"""Training Complete!
            
Training Results:
• Training R²: {results['train_r2']:.4f}
• Testing R²: {results['test_r2']:.4f}
• Training Cost: {results['train_cost']:.4f}
• Testing Cost: {results['test_cost']:.4f}"""
            
            self.training_status.text = status_text
            
            # Update performance tab
            self.performance_metrics.text = status_text
            
            # Enable predict button
            self.predict_button.disabled = False
            
        except Exception as e:
            self.training_status.text = f"Training failed: {str(e)}"
        
        finally:
            # Re-enable train button
            self.train_button.disabled = False
            self.train_button.text = "Start Training"
    
    def make_prediction(self, button):
        """Make a prediction using current input values"""
        if not self.model_trained:
            self.show_popup("Error", "Please train the model first")
            return
        
        try:
            # Get input values from sliders
            input_values = []
            feature_order = ['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water',
                           'Superplasticizer', 'CoarseAggregate', 'FineAggregate', 'Age']
            
            for feature in feature_order:
                input_values.append(self.input_sliders[feature].value)
            
            input_array = np.array(input_values)
            
            # Make prediction
            prediction = self.model.predict(input_array)
            
            # Classify strength
            if prediction < 20:
                strength_class = "Low Strength"
                color_indicator = "🔴"
            elif prediction < 40:
                strength_class = "Medium Strength"
                color_indicator = "🟡"
            elif prediction < 60:
                strength_class = "High Strength"
                color_indicator = "🟢"
            else:
                strength_class = "Very High Strength"
                color_indicator = "🟢"
            
            result_text = f"""Predicted Strength: {prediction:.2f} MPa
            
Classification: {color_indicator} {strength_class}

This prediction is based on the input concrete mix proportions."""
            
            self.prediction_result.text = result_text
            
        except Exception as e:
            self.prediction_result.text = f"Prediction failed: {str(e)}"
    
    def save_model(self, button):
        """Save the trained model"""
        if not self.model_trained:
            self.show_popup("Error", "No trained model to save")
            return
        
        try:
            self.model.save_model('concrete_model_mobile.npz')
            self.show_popup("Success", "Model saved successfully!")
        except Exception as e:
            self.show_popup("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self, button):
        """Load a saved model"""
        try:
            if self.model.load_model('concrete_model_mobile.npz'):
                self.model_trained = True
                self.predict_button.disabled = False
                self.show_popup("Success", "Model loaded successfully!")
            else:
                self.show_popup("Error", "No saved model found")
        except Exception as e:
            self.show_popup("Error", f"Failed to load model: {str(e)}")
    
    def show_popup(self, title, message):
        """Show a popup message"""
        popup_layout = MDBoxLayout(
            orientation='vertical',
            padding=dp(20),
            spacing=dp(15)
        )
        
        popup_label = MDLabel(
            text=message,
            size_hint_y=None,
            height=dp(100)
        )
        popup_layout.add_widget(popup_label)
        
        close_button = MDRaisedButton(
            text="OK",
            size_hint_y=None,
            height=dp(40)
        )
        popup_layout.add_widget(close_button)
        
        popup = Popup(
            title=title,
            content=popup_layout,
            size_hint=(0.8, 0.4)
        )
        
        close_button.bind(on_release=popup.dismiss)
        popup.open()

if __name__ == '__main__':
    ConcreteStrengthApp().run()